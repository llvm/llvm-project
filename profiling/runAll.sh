#!/bin/bash

projectRoot="/home/muyao/projects/llvm-project"
testcaseOutDir="${projectRoot}/profiling/out"

# Compile with normal openmp workdistribute
for file in ${testcaseOutDir}/*.out; do
  if [ -f "${file}" ]; then
    LD_LIBRARY_PATH=${projectRoot}/build/lib:${projectRoot}/build/runtimes/runtimes-bins/openmp/runtime/src:${projectRoot}/../xla_glue_runtime/third_party/libs \
      LD_PRELOAD=${projectRoot}/../xla_glue_runtime/build/libjit-code-executor.so \
      LIBOMPTARGET_INFO=-1 LIBOMPTARGET_DEBUG=1 \
      ${file}
  fi
done

