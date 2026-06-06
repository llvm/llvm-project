// RUN: %libomptarget-compile-nvptx64-nvidia-cuda -I %S
// RUN: env LIBOMPTARGET_NEXTGEN_PLUGINS=1 \
// RUN:   %libomptarget-run-nvptx64-nvidia-cuda | %fcheck-nvptx64-nvidia-cuda
// REQUIRES: nvptx64-nvidia-cuda

#include "omp_dynamic_shared_memory_mixed.inc"
// CHECK: PASS
