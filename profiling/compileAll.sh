#!/bin/bash

projectRoot="/home/muyao/projects/llvm-project"
testcaseDir="${projectRoot}/profiling/testcases"
testcaseOutDir="${projectRoot}/profiling/out"

# Clean the out dir
rm "${testcaseOutDir}"/*\.out

# Compile with normal openmp workdistribute
echo "Compiling normal test cases in dir: ${testcaseDir}"
for file in ${testcaseDir}/*.f90; do
  if [ -f "${file}" ]; then
    filename="$(echo "$(basename "${file}")" | sed "s/\..*//g")"

    ${projectRoot}/build/bin/flang -O3 -fopenmp -fopenmp-version=60 \
      -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_80 \
      -I${projectRoot}/build/runtimes/runtimes-bins/openmp/runtime/src \
      -L${projectRoot}/build/runtimes/runtimes-bins/openmp/runtime/src \
      -L${projectRoot}/build/lib \
      -lomptarget \
      ${file} -o "${testcaseOutDir}/${filename}-omp.out" "$@"
  fi
done

# Compile with XLA
echo "Compiling XLA test cases in dir: ${testcaseDir}"
for file in ${testcaseDir}/*.f90; do
  if [ -f "${file}" ]; then
    filename="$(echo "$(basename "${file}")" | sed "s/\..*//g")"

    ${projectRoot}/build/bin/flang -O3 -fopenmp -fopenmp-version=60 \
      -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_80 \
      -I${projectRoot}/build/runtimes/runtimes-bins/openmp/runtime/src \
      -L${projectRoot}/build/runtimes/runtimes-bins/openmp/runtime/src \
      -L${projectRoot}/build/lib \
      -lomptarget \
      ${file} -o "${testcaseOutDir}/${filename}-xla.out" -mmlir --jit-workdistribute "$@"
  fi
done

