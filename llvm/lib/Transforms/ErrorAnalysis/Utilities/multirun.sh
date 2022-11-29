#!/bin/bash

src_directory=$(dirname $1)
cd $src_directory || exit

make build_pass NO_DATA_DUMP=-DNO_DATA_DUMP MEMORY_OPT=-DMEMORY_OPT

input_values=(-2.9e-06, -1.9e-06, -9e-07, 1e-07, 1.1e-06, 2.1e-06, 3.1e-06)

for ((i=0;i<${#input_values[@]};i++)); do
  echo ${input_values[i]} | make run_pass2
done