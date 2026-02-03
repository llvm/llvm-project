#!/bin/bash

projectRoot="/home/muyao/projects/llvm-project"
testcaseOutDir="${projectRoot}/profiling/out"
profilingResDir="${projectRoot}/profiling/results"

# Compile with normal openmp workdistribute
for file in ${testcaseOutDir}/*.out; do
  if [ -f "${file}" ]; then
    filename=$(echo $(basename ${file})| sed "s/\..*//g")
    echo "Filename: ${filename}"

    LD_LIBRARY_PATH=${projectRoot}/build/lib:${projectRoot}/build/runtimes/runtimes-bins/openmp/runtime/src:${projectRoot}/../xla_glue_runtime/third_party/libs \
      LD_PRELOAD="${projectRoot}/../xla_glue_runtime/build/libjit-code-executor.so" \
      LIBOMPTARGET_INFO=0 LIBOMPTARGET_DEBUG=0 \
      nsys profile -t cuda -o "${profilingResDir}/${filename}" ${file}
  fi
done

