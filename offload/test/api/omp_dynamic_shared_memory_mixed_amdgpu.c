// RUN: %libomptarget-compile-amdgcn-amd-amdhsa -O1 -mllvm -openmp-opt-inline-device -I %S
// RUN: env LIBOMPTARGET_NEXTGEN_PLUGINS=1 \
// RUN:   %libomptarget-run-amdgcn-amd-amdhsa | %fcheck-amdgcn-amd-amdhsa
// REQUIRES: amdgcn-amd-amdhsa

#include "omp_dynamic_shared_memory_mixed.inc"
// CHECK: PASS
