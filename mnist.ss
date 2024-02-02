#!/usr/bin/scheme --script

(let ()
  (import (mnist-loading)
          (learning)
          (cmatrices)
          (utils)
          (chezscheme))

  ;; enlarge the allocate block each time and avoid collecting frequently
  ;; -- it speeds things up when we are using native matrices

  ;; (collect-trip-bytes (* 10 (collect-trip-bytes)))
  ;; (collect-generation-radix 1000000)

  ;; try to continue from saved file
  (let* ((network (initialize-network '(784 30 30 10)))
         (training-data (load-training-data))
         (testing-data (load-testing-data))
         (clen (length (command-line)))
         (minibatch-size (or (parse-cmdline-number "--minibatch-size" "-m") 10))
         (epochs (or (parse-cmdline-number "--epochs" "-e") 2))
         (learning-rate (or (parse-cmdline-number "--learning-rate") 3.0))
         (trained-network
          (time (stochastic-gradient-descent
                 network
                 training-data
                 testing-data
                 minibatch-size
                 epochs
                 learning-rate
                 ))))

    trained-network))
