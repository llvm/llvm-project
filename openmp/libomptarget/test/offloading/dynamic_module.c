// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu -DSHARED -fPIC -shared -o %t.so && %libomptarget-compile-aarch64-unknown-linux-gnu %t.so && %libomptarget-run-aarch64-unknown-linux-gnu 2>&1 | %fcheck-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu -DSHARED -fPIC -shared -o %t.so && %libomptarget-compile-powerpc64-ibm-linux-gnu %t.so && %libomptarget-run-powerpc64-ibm-linux-gnu 2>&1 | %fcheck-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu -DSHARED -fPIC -shared -o %t.so && %libomptarget-compile-powerpc64le-ibm-linux-gnu %t.so && %libomptarget-run-powerpc64le-ibm-linux-gnu 2>&1 | %fcheck-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compile-x86_64-pc-linux-gnu -DSHARED -fPIC -shared -o %t.so && %libomptarget-compile-x86_64-pc-linux-gnu %t.so && %libomptarget-run-x86_64-pc-linux-gnu 2>&1 | %fcheck-x86_64-pc-linux-gnu
// RUN: %libomptarget-compile-nvptx64-nvidia-cuda -DSHARED -fPIC -shared -o %t.so && %libomptarget-compile-nvptx64-nvidia-cuda %t.so && %libomptarget-run-nvptx64-nvidia-cuda 2>&1 | %fcheck-nvptx64-nvidia-cuda

#ifdef SHARED
void foo() {}
#else
#include <stdio.h>
int main() {
#pragma omp target
  ;
  // CHECK: DONE.
  printf("%s\n", "DONE.");
  return 0;
}
#endif
