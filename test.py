# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================


from resnet import *
from datetime import datetime
import time
from input import *
import pandas as pd



class Train(object):
    '''
    This Object is responsible for all the training and validation process
    '''
    def __init__(self):
        # Set up all the placeholders
        self.placeholders()


    def placeholders(self):
        '''
        There are five placeholders in total.
        image_placeholder and label_placeholder are for train images and labels
        vali_image_placeholder and vali_label_placeholder are for validation imgaes and labels
        lr_placeholder is for learning rate. Feed in learning rate each time of training
        implements learning rate decay easily
        '''
        self.image_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.train_batch_size, IMG_HEIGHT,
                                                        IMG_WIDTH, IMG_DEPTH])
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size])

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,
                                                                IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.vali_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[FLAGS.validation_batch_size])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])



    def build_train_validation_graph(self):
        '''
        This function builds the train graph and validation graph at the same time.
        
        '''
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        logits = inference(self.image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        vali_logits = inference(self.vali_image_placeholder, FLAGS.num_residual_blocks, reuse=True)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = self.loss(logits, self.label_placeholder)
        self.full_loss = tf.add_n([loss] + regu_losses)

        predictions = tf.nn.softmax(logits)
        self.train_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)


        # Validation loss
        self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        vali_predictions = tf.nn.softmax(vali_logits)
        self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)

        self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss,
                                                                self.train_top1_error)
        self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)



    def train(self):
        '''
        This is the main function for training
        '''

        # For the first step, we are loading all training images and validation images into the
        # memory
        # all_data, all_labels = prepare_train_data(padding_size=FLAGS.padding_size)
        # vali_data, vali_labels = prepare_vali_data(padding_size=FLAGS.padding_size)

        # Build the graph for train and validation
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        with tf.device('/gpu:0'):
            with tf.Session(config=tfconfig) as sess:
                with sess.graph.as_default():

                    self.build_train_validation_graph()

                    # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
                    # summarizing operations by running summary_op. Initialize a new session
                    saver = tf.train.Saver(tf.global_variables())
                    summary_op = tf.summary.merge_all()
                    init = tf.initialize_all_variables()
                    # sess = tf.Session()


                    # If you want to load from a checkpoint
                    if FLAGS.is_use_ckpt is True:
                        saver.restore(sess, FLAGS.ckpt_path)
                        print('Restored from checkpoint...')
                    else:
                        sess.run(init)

                    # This summary writer object helps write summaries on tensorboard
                    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)


                    # These lists are used to save a csv file at last
                    step_list = []
                    train_error_list = []
                    val_error_list = []

                    print ('Start training...')
                    print ('----------------------------')

                    train_list = [n for n in range(0,3804)]
                    vali_list = [n for n in range(0,800)]
                    train_list_array = np.array(train_list)
                    vali_list_array = np.array(vali_list)

                    order = np.random.permutation(3804)
                    train_list_array = train_list_array[order]
                    order = np.random.permutation(800)
                    vali_list_array = vali_list_array[order]

                    all_labels = get_train_lable(train_dir)
                    vali_labels = get_train_lable(vali_dir)



                    for step in range(FLAGS.train_steps):

                        train_batch_data, train_batch_labels = self.generate_augment_train_batch(train_list_array, all_labels,
                                                                                    FLAGS.train_batch_size)


                        validation_batch_data, validation_batch_labels = self.generate_vali_batch(vali_list_array,
                                                                       vali_labels, FLAGS.validation_batch_size)



                        # Want to validate once before training. You may check the theoretical validation
                        # loss first
                        if step % FLAGS.report_freq == 0:

                            if FLAGS.is_full_validation is True:
                                validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss,
                                                        top1_error=self.vali_top1_error, vali_data=vali_data,
                                                        vali_labels=vali_labels, session=sess,
                                                        batch_data=train_batch_data, batch_label=train_batch_labels)

                                vali_summ = tf.Summary()
                                vali_summ.value.add(tag='full_validation_error',
                                                    simple_value=validation_error_value.astype(np.float))
                                summary_writer.add_summary(vali_summ, step)
                                summary_writer.flush()

                            else:
                                _, validation_error_value, validation_loss_value = sess.run([self.val_op,
                                                                                 self.vali_top1_error,
                                                                             self.vali_loss],
                                                            {self.image_placeholder: train_batch_data,
                                                             self.label_placeholder: train_batch_labels,
                                                             self.vali_image_placeholder: validation_batch_data,
                                                             self.vali_label_placeholder: validation_batch_labels,
                                                             self.lr_placeholder: FLAGS.init_lr})

                            val_error_list.append(validation_error_value)


                        start_time = time.time()

                        _, _, train_loss_value, train_error_value = sess.run([self.train_op, self.train_ema_op,
                                                                       self.full_loss, self.train_top1_error],
                                            {self.image_placeholder: train_batch_data,
                                              self.label_placeholder: train_batch_labels,
                                              self.vali_image_placeholder: validation_batch_data,
                                              self.vali_label_placeholder: validation_batch_labels,
                                              self.lr_placeholder: FLAGS.init_lr})
                        duration = time.time() - start_time


                        if step % FLAGS.report_freq == 0:
                            summary_str = sess.run(summary_op, {self.image_placeholder: train_batch_data,
                                                                self.label_placeholder: train_batch_labels,
                                                                self.vali_image_placeholder: validation_batch_data,
                                                                self.vali_label_placeholder: validation_batch_labels,
                                                                self.lr_placeholder: FLAGS.init_lr})
                            summary_writer.add_summary(summary_str, step)

                            num_examples_per_step = FLAGS.train_batch_size
                            examples_per_sec = num_examples_per_step / duration
                            sec_per_batch = float(duration)

                            format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                            print (format_str % (datetime.now(), step, train_loss_value, examples_per_sec,
                                                sec_per_batch))
                            print ('Train top1 error = ', train_error_value)
                            print ('Validation top1 error = %.4f' % validation_error_value)
                            print ('Validation loss = ', validation_loss_value)
                            print ('----------------------------')

                            step_list.append(step)
                            train_error_list.append(train_error_value)



                        if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
                            FLAGS.init_lr = 0.1 * FLAGS.init_lr
                            print ('Learning rate decayed to ', FLAGS.init_lr)

                        # Save checkpoints every 10000 steps
                        if step % 2000 == 0 or (step + 1) == FLAGS.train_steps:
                            checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                            saver.save(sess, checkpoint_path, global_step=step)

                            df = pd.DataFrame(data={'step':step_list, 'train_error':train_error_list,
                                            'validation_error': val_error_list})
                            df.to_csv(train_dir + FLAGS.version + '_error.csv')


    def test(self):
        '''
        This function is used to evaluate the test data. Please finish pre-precessing in advance

        :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
        img_depth]
        :return: the softmax probability with shape [num_test_images, num_labels]
        '''

        train_list = [n for n in range(0, 3804)]
        vali_list = [n for n in range(0, 160)]
        train_list_array = np.array(train_list)
        vali_list_array = np.array(vali_list)

        order = np.random.permutation(3804)
        train_list_array = train_list_array[order]
        order = np.random.permutation(160)
        vali_list_array = vali_list_array[order]

        all_labels = get_train_lable(train_dir)
        vali_labels = get_train_lable(vali_dir)

        test_data, test_labels = self.generate_vali_batch_my(vali_list_array, vali_labels,
                                                                                    FLAGS.test_batch_size)

        test_image_array = test_data
        num_test_images = len(test_image_array)
        num_batches = num_test_images // FLAGS.test_batch_size
        remain_images = num_test_images % FLAGS.test_batch_size
        print ('%i test batches in total...' %num_batches)

        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                        IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        # Build the test graph
        logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        predictions = tf.nn.softmax(logits)

        # Initialize a new session and restore a checkpoint
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()

        saver.restore(sess, FLAGS.test_ckpt_path)
        print ('Model restored from ', FLAGS.test_ckpt_path)

        prediction_array = np.array([]).reshape(-1, NUM_CLASS)
        # Test by batches
        for step in range(num_batches):
            if step % 10 == 0:
                print ('%i batches finished!' %step)
            offset = step * FLAGS.test_batch_size
            test_image_batch = test_image_array[offset:offset+FLAGS.test_batch_size, ...]

            batch_prediction_array = sess.run(predictions,
                                        feed_dict={self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        # If test_batch_size is not a divisor of num_test_images
        if remain_images != 0:
            self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                        IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
            # Build the test graph
            logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=True)
            predictions = tf.nn.softmax(logits)

            test_image_batch = test_image_array[-remain_images:, ...]

            batch_prediction_array = sess.run(predictions, feed_dict={
                self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        print(test_labels)
        print(prediction_array)
        ok = 0
        for i in range(FLAGS.test_size):
            if test_labels[i] == 0:
                if prediction_array[i][0] >= prediction_array[i][1]:
                    ok = ok + 1
            else:
                if prediction_array[i][0] <= prediction_array[i][1]:
                    ok = ok + 1
        error = float(FLAGS.test_size - ok) / float(FLAGS.test_size)
        print("error is : %f" % error)
        return prediction_array



    ## Helper functions
    def loss(self, logits, labels):
        '''
        Calculate the cross entropy loss given logits and true labels
        :param logits: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size]
        :return: loss tensor with shape [1]
        '''
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean


    def top_k_error(self, predictions, labels, k):
        '''
        Calculate the top-k error
        :param predictions: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k: int
        :return: tensor with shape [1]
        '''
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)

    def generate_vali_batch_my(self, vali_data, vali_label, vali_batch_size):
        '''
        If you want to use a random batch of validation data to validate instead of using the
        whole validation data, this function helps you generate that batch
        :param vali_data: 4D numpy array
        :param vali_label: 1D numpy array
        :param vali_batch_size: int
        :return: 4D numpy array and 1D numpy array
        '''

        # batch_label = []
        # offset = np.random.choice(vali_size - vali_batch_size, 1)[0]
        # batch_data = vali_data[offset:offset+vali_batch_size]
        # h=0
        # data = np.zeros(shape=(vali_batch_size, 512, 512, 3))
        # for i in batch_data:
        #     j = int(i % 200) + 1
        #     k = int(i / 200)
        #
        #     if j < 10:
        #         img_name = vali_dir + "/000" + str(j) + ".PNG"
        #     elif 10 <= j < 100:
        #         img_name = vali_dir + "/00" + str(j) + ".PNG"
        #     elif 100 <= j < 1000:
        #         img_name = vali_dir + "/0" + str(j) + ".PNG"
        #     else:
        #         img_name = vali_dir + "/" + str(j) + ".PNG"
        #     # print(batch_data)
        #     # print(i)
        #     # print(img_name)
        #
        #     img = cv2.imread(img_name, 0)
        #     # img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        #     imgrgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #     # imgrgb = imgrgb.astype(np.float32, copy=False)
        #     if k == 0:
        #         imgarray = imgrgb[np.newaxis, :]
        #     elif k == 1:
        #         imgrgb = imgrgb[:, ::-1, :]
        #         imgarray = imgrgb[np.newaxis, :]
        #     elif k == 2:
        #         imgrgb = imgrgb[::-1, :, :]
        #         imgarray = imgrgb[np.newaxis, :]
        #     else:
        #         imgrgb = imgrgb[:, ::-1, :]
        #         imgrgb = imgrgb[::-1, :, :]
        #         imgarray = imgrgb[np.newaxis, :]
        #
        #     if h == 0:
        #         imgarrayall = imgarray
        #     else:
        #         imgarrayall = np.concatenate((imgarrayall, imgarray), 0)
        #
        #     if h == (vali_batch_size - 1):
        #         data = imgarrayall
        #
        #
        #     batch_label.append(vali_label[int(i % 200)])
        #     h = h + 1
        #
        # batch_data = data.astype(np.float32, copy=False)

        go = True

        batch_label_end = []
        data_end = np.zeros(shape=(vali_batch_size, 512, 512, 3))
        a_i = 0
        b_i = 0
        name_end = []
        while go:
            name = []
            batch_label = []
            offset = np.random.choice(160 - vali_batch_size, 1)[0]
            batch_data = vali_data[offset:offset+vali_batch_size]
            h=0
            data = np.zeros(shape=(vali_batch_size, 512, 512, 3))
            for i in batch_data:
                j = int(i % 40) + 1
                k = int(i / 40)

                if j < 10:
                    img_name = vali_dir + "/000" + str(j) + ".PNG"
                elif 10 <= j < 100:
                    img_name = vali_dir + "/00" + str(j) + ".PNG"
                elif 100 <= j < 1000:
                    img_name = vali_dir + "/0" + str(j) + ".PNG"
                else:
                    img_name = vali_dir + "/" + str(j) + ".PNG"
                # print(batch_data)
                # print(i)
                # print(img_name)
                name.append(img_name)

                img = cv2.imread(img_name, 0)
                # img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
                imgrgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # imgrgb = imgrgb.astype(np.float32, copy=False)
                if k == 0:
                    imgarray = imgrgb[np.newaxis, :]
                elif k == 1:
                    imgrgb = imgrgb[:, ::-1, :]
                    imgarray = imgrgb[np.newaxis, :]
                elif k == 2:
                    imgrgb = imgrgb[::-1, :, :]
                    imgarray = imgrgb[np.newaxis, :]
                else:
                    imgrgb = imgrgb[:, ::-1, :]
                    imgrgb = imgrgb[::-1, :, :]
                    imgarray = imgrgb[np.newaxis, :]

                if h == 0:
                    imgarrayall = imgarray
                else:
                    imgarrayall = np.concatenate((imgarrayall, imgarray), 0)

                if h == (vali_batch_size - 1):
                    data = imgarrayall


                batch_label.append(vali_label[int(i % 40)])
                h = h + 1

            # batch_data = data.astype(np.float32, copy=False)
            # print(batch_label)
            a = []
            b = []
            for i in range(vali_batch_size):
                if batch_label[i] == 0:
                    a.append(i)
                else:
                    b.append(i)
            # print(a)
            # print(b)
            key = vali_batch_size / 2
            if len(b) == 0:
                go = True
            elif vali_batch_size > len(b) > 0:
                for j in range(len(a)):
                    if a_i < key:
                        batch_label_end.append(batch_label[int(a[j])])
                        name_end.append(name[int(a[j])])
                        if a_i == 0:
                            data_end = data[a[j]][np.newaxis, :]
                        else:
                            data_end = np.concatenate((data_end, data[a[j]][np.newaxis, :]), 0)
                        a_i = a_i + 1
                for j in range(len(b)):
                    if b_i < key:
                        name_end.append(name[int(b[j])])

                        batch_label_end.append(batch_label[int(b[j])])
                        data_end = np.concatenate((data_end, data[b[j]][np.newaxis, :]), 0)
                        b_i = b_i + 1
            else:
                go = True
            # print(a_i,  b_i)
            if a_i == b_i == key:
                go = False

        batch_data = data_end.astype(np.float32, copy=False)


        batch_data = whitening_image(batch_data)
        # offset = np.random.choice(EPOCH_SIZE - vali_batch_size, 1)[0]
        # vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
        # vali_label_batch = vali_label[offset:offset+vali_batch_size]

        print(name_end)
        return batch_data, batch_label_end



    def generate_vali_batch(self, vali_data, vali_label, vali_batch_size):
        '''
        If you want to use a random batch of validation data to validate instead of using the
        whole validation data, this function helps you generate that batch
        :param vali_data: 4D numpy array
        :param vali_label: 1D numpy array
        :param vali_batch_size: int
        :return: 4D numpy array and 1D numpy array
        '''

        # batch_label = []
        # offset = np.random.choice(vali_size - vali_batch_size, 1)[0]
        # batch_data = vali_data[offset:offset+vali_batch_size]
        # h=0
        # data = np.zeros(shape=(vali_batch_size, 512, 512, 3))
        # for i in batch_data:
        #     j = int(i % 200) + 1
        #     k = int(i / 200)
        #
        #     if j < 10:
        #         img_name = vali_dir + "/000" + str(j) + ".PNG"
        #     elif 10 <= j < 100:
        #         img_name = vali_dir + "/00" + str(j) + ".PNG"
        #     elif 100 <= j < 1000:
        #         img_name = vali_dir + "/0" + str(j) + ".PNG"
        #     else:
        #         img_name = vali_dir + "/" + str(j) + ".PNG"
        #     # print(batch_data)
        #     # print(i)
        #     # print(img_name)
        #
        #     img = cv2.imread(img_name, 0)
        #     # img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        #     imgrgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #     # imgrgb = imgrgb.astype(np.float32, copy=False)
        #     if k == 0:
        #         imgarray = imgrgb[np.newaxis, :]
        #     elif k == 1:
        #         imgrgb = imgrgb[:, ::-1, :]
        #         imgarray = imgrgb[np.newaxis, :]
        #     elif k == 2:
        #         imgrgb = imgrgb[::-1, :, :]
        #         imgarray = imgrgb[np.newaxis, :]
        #     else:
        #         imgrgb = imgrgb[:, ::-1, :]
        #         imgrgb = imgrgb[::-1, :, :]
        #         imgarray = imgrgb[np.newaxis, :]
        #
        #     if h == 0:
        #         imgarrayall = imgarray
        #     else:
        #         imgarrayall = np.concatenate((imgarrayall, imgarray), 0)
        #
        #     if h == (vali_batch_size - 1):
        #         data = imgarrayall
        #
        #
        #     batch_label.append(vali_label[int(i % 200)])
        #     h = h + 1
        #
        # batch_data = data.astype(np.float32, copy=False)

        go = True

        batch_label_end = []
        data_end = np.zeros(shape=(vali_batch_size, 512, 512, 3))
        a_i = 0
        b_i = 0
        name_end = []
        while go:
            name = []
            batch_label = []
            offset = np.random.choice(vali_size - vali_batch_size, 1)[0]
            batch_data = vali_data[offset:offset+vali_batch_size]
            h=0
            data = np.zeros(shape=(vali_batch_size, 512, 512, 3))
            for i in batch_data:
                j = int(i % 200) + 1
                k = int(i / 200)

                if j < 10:
                    img_name = vali_dir + "/000" + str(j) + ".PNG"
                elif 10 <= j < 100:
                    img_name = vali_dir + "/00" + str(j) + ".PNG"
                elif 100 <= j < 1000:
                    img_name = vali_dir + "/0" + str(j) + ".PNG"
                else:
                    img_name = vali_dir + "/" + str(j) + ".PNG"
                # print(batch_data)
                # print(i)
                # print(img_name)
                name.append(img_name)

                img = cv2.imread(img_name, 0)
                # img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
                imgrgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # imgrgb = imgrgb.astype(np.float32, copy=False)
                if k == 0:
                    imgarray = imgrgb[np.newaxis, :]
                elif k == 1:
                    imgrgb = imgrgb[:, ::-1, :]
                    imgarray = imgrgb[np.newaxis, :]
                elif k == 2:
                    imgrgb = imgrgb[::-1, :, :]
                    imgarray = imgrgb[np.newaxis, :]
                else:
                    imgrgb = imgrgb[:, ::-1, :]
                    imgrgb = imgrgb[::-1, :, :]
                    imgarray = imgrgb[np.newaxis, :]

                if h == 0:
                    imgarrayall = imgarray
                else:
                    imgarrayall = np.concatenate((imgarrayall, imgarray), 0)

                if h == (vali_batch_size - 1):
                    data = imgarrayall


                batch_label.append(vali_label[int(i % 200)])
                h = h + 1

            # batch_data = data.astype(np.float32, copy=False)
            # print(batch_label)
            a = []
            b = []
            for i in range(vali_batch_size):
                if batch_label[i] == 0:
                    a.append(i)
                else:
                    b.append(i)
            # print(a)
            # print(b)
            key = vali_batch_size / 2
            if len(b) == 0:
                go = True
            elif vali_batch_size > len(b) > 0:
                for j in range(len(a)):
                    if a_i < key:
                        batch_label_end.append(batch_label[int(a[j])])
                        name_end.append(name[int(a[j])])
                        if a_i == 0:
                            data_end = data[a[j]][np.newaxis, :]
                        else:
                            data_end = np.concatenate((data_end, data[a[j]][np.newaxis, :]), 0)
                        a_i = a_i + 1
                for j in range(len(b)):
                    if b_i < key:
                        name_end.append(name[int(b[j])])

                        batch_label_end.append(batch_label[int(b[j])])
                        data_end = np.concatenate((data_end, data[b[j]][np.newaxis, :]), 0)
                        b_i = b_i + 1
            else:
                go = True
            # print(a_i,  b_i)
            if a_i == b_i == key:
                go = False

        batch_data = data_end.astype(np.float32, copy=False)


        batch_data = whitening_image(batch_data)
        # offset = np.random.choice(EPOCH_SIZE - vali_batch_size, 1)[0]
        # vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
        # vali_label_batch = vali_label[offset:offset+vali_batch_size]

        print(name_end)
        return batch_data, batch_label_end


    def generate_augment_train_batch(self, train_data, train_labels, train_batch_size):
        '''
        This function helps generate a batch of train data, and random crop, horizontally flip
        and whiten them at the same time
        :param train_data: 4D numpy array
        :param train_labels: 1D numpy array
        :param train_batch_size: int
        :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
        '''
        go = True

        batch_label_end = []
        data_end = np.zeros(shape=(train_batch_size, 512, 512, 3))
        a_i = 0
        b_i = 0
        name_end=[]
        while go:
            name=[]
            batch_label = []
            data = np.zeros(shape=(train_batch_size, 512, 512, 3))
            offset = np.random.choice(train_size - train_batch_size, 1)[0]
            batch_data = train_data[offset:offset+train_batch_size]
            h=0

            for i in batch_data:
                j = int(i % 951) + 200
                k = int(i / 951)
                if j < 10:
                    img_name = train_dir + "/000" + str(j) + ".PNG"
                elif 10 <= j < 100:
                    img_name = train_dir + "/00" + str(j) + ".PNG"
                elif 100 <= j < 1000:
                    img_name = train_dir + "/0" + str(j) + ".PNG"
                else:
                    img_name = train_dir + "/" + str(j) + ".PNG"
                name.append(img_name)
                # print(batch_data)
                # print(i)
                # print(img_name)


                img = cv2.imread(img_name, 0)
                # img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
                imgrgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # imgrgb = imgrgb.astype(np.float32, copy=False)
                if k == 0:
                    imgarray = imgrgb[np.newaxis, :]
                elif k == 1:
                    imgrgb = imgrgb[:, ::-1, :]
                    imgarray = imgrgb[np.newaxis, :]
                elif k == 2:
                    imgrgb = imgrgb[::-1, :, :]
                    imgarray = imgrgb[np.newaxis, :]
                else:
                    imgrgb = imgrgb[:, ::-1, :]
                    imgrgb = imgrgb[::-1, :, :]
                    imgarray = imgrgb[np.newaxis, :]

                if h == 0:
                    imgarrayall = imgarray
                else:
                    imgarrayall = np.concatenate((imgarrayall, imgarray), 0)

                if h == (train_batch_size - 1):
                    data = imgarrayall


                batch_label.append(train_labels[int(i % 951)])
                h = h + 1
            # print(batch_label)
            a = []
            b = []
            for i in range(train_batch_size):
                if batch_label[i] == 0:
                    a.append(i)
                else:
                    b.append(i)
            # print(a)
            # print(b)
            key = train_batch_size/2
            if len(b) == 0:
                go = True
            elif  train_batch_size > len(b) > 0:
                for j in range(len(a)):
                    if a_i < key:
                        batch_label_end.append(batch_label[int(a[j])])
                        name_end.append(name[int(a[j])])
                        if a_i == 0:
                            data_end = data[a[j]][np.newaxis, :]
                        else:
                            data_end = np.concatenate((data_end, data[a[j]][np.newaxis, :]), 0)
                        a_i = a_i +1
                for j in range(len(b)):
                    if b_i < key:
                        name_end.append(name[int(b[j])])
                        batch_label_end.append(batch_label[int(b[j])])
                        data_end = np.concatenate((data_end, data[b[j]][np.newaxis, :]), 0)
                        b_i = b_i + 1
            else:
                go = True
            # print(a_i,  b_i)
            if a_i == b_i == key:
                go = False

        batch_data = data_end.astype(np.float32, copy=False)

        pad_width = ((0, 0), (FLAGS.padding_size, FLAGS.padding_size), (FLAGS.padding_size, FLAGS.padding_size), (0, 0))
        batch_data = np.pad(batch_data, pad_width=pad_width, mode='constant', constant_values=0)

        batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)

        batch_data = whitening_image(batch_data)

        # batch_label = train_labels[offset:offset+FLAGS.train_batch_size]
        print(name_end)


        return batch_data, batch_label_end



    def train_operation(self, global_step, total_loss, top1_error):
        '''
        Defines train operations
        :param global_step: tensor variable with shape [1]
        :param total_loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :return: two operations. Running train_op will do optimization once. Running train_ema_op
        will generate the moving average of train error and train loss for tensorboard
        '''
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', total_loss)
        tf.summary.scalar('train_top1_error', top1_error)

        # The ema object help calculate the moving average of train loss and train error
        ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])
        tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
        tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op


    def validation_op(self, validation_step, top1_error, loss):
        '''
        Defines validation operations
        :param validation_step: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param loss: tensor with shape [1]
        :return: validation operation
        '''

        # This ema object help calculate the moving average of validation loss and error

        # ema with decay = 0.0 won't average things at all. This returns the original error
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)


        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        top1_error_val = ema.average(top1_error)
        top1_error_avg = ema2.average(top1_error)
        loss_val = ema.average(loss)
        loss_val_avg = ema2.average(loss)

        # Summarize these values on tensorboard
        tf.summary.scalar('val_top1_error', top1_error_val)
        tf.summary.scalar('val_top1_error_avg', top1_error_avg)
        tf.summary.scalar('val_loss', loss_val)
        tf.summary.scalar('val_loss_avg', loss_val_avg)
        return val_op


    def full_validation(self, loss, top1_error, session, vali_data, vali_labels, batch_data,
                        batch_label):
        '''
        Runs validation on all the 10000 valdiation images
        :param loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param session: the current tensorflow session
        :param vali_data: 4D numpy array
        :param vali_labels: 1D numpy array
        :param batch_data: 4D numpy array. training batch to feed dict and fetch the weights
        :param batch_label: 1D numpy array. training labels to feed the dict
        :return: float, float
        '''
        num_batches = 10000 // FLAGS.validation_batch_size
        order = np.random.choice(10000, num_batches * FLAGS.validation_batch_size)
        vali_data_subset = vali_data[order, ...]
        vali_labels_subset = vali_labels[order]

        loss_list = []
        error_list = []

        for step in range(num_batches):
            offset = step * FLAGS.validation_batch_size
            feed_dict = {self.image_placeholder: batch_data, self.label_placeholder: batch_label,
                self.vali_image_placeholder: vali_data_subset[offset:offset+FLAGS.validation_batch_size, ...],
                self.vali_label_placeholder: vali_labels_subset[offset:offset+FLAGS.validation_batch_size],
                self.lr_placeholder: FLAGS.init_lr}
            loss_value, top1_error_value = session.run([loss, top1_error], feed_dict=feed_dict)
            loss_list.append(loss_value)
            error_list.append(top1_error_value)

        return np.mean(loss_list), np.mean(error_list)


# maybe_download_and_extract()
# Initialize the Train object
train = Train()
# Start the training session
train.test()




