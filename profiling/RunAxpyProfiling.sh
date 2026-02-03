LD_LIBRARY_PATH=./build/lib:./build/runtimes/runtimes-bins/openmp/runtime/src:../xla_glue_runtime/third_party/libs \
  LD_PRELOAD=../xla_glue_runtime/build/libjit-code-executor.so LIBOMPTARGET_INFO=-1 LIBOMPTARGET_DEBUG=1  \
  nsys profile -t cuda -o ./testprofile ./a.out
