// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu -DSHARED -fPIC -shared -o %t.so && %clang %flags %s -o %t-aarch64-unknown-linux-gnu -ldl && %libomptarget-run-aarch64-unknown-linux-gnu %t.so 2>&1 | %fcheck-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu -DSHARED -fPIC -shared -o %t.so && %clang %flags %s -o %t-powerpc64-ibm-linux-gnu -ldl && %libomptarget-run-powerpc64-ibm-linux-gnu %t.so 2>&1 | %fcheck-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu -DSHARED -fPIC -shared -o %t.so && %clang %flags %s -o %t-powerpc64le-ibm-linux-gnu -ldl && %libomptarget-run-powerpc64le-ibm-linux-gnu %t.so 2>&1 | %fcheck-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compile-x86_64-pc-linux-gnu -DSHARED -fPIC -shared -o %t.so && %clang %flags %s -o %t-x86_64-pc-linux-gnu -ldl && %libomptarget-run-x86_64-pc-linux-gnu %t.so 2>&1 | %fcheck-x86_64-pc-linux-gnu
// RUN: %libomptarget-compile-nvptx64-nvidia-cuda -DSHARED -fPIC -shared -o %t.so && %clang %flags %s -o %t-nvptx64-nvidia-cuda -ldl && %libomptarget-run-nvptx64-nvidia-cuda %t.so 2>&1 | %fcheck-nvptx64-nvidia-cuda

#ifdef SHARED
#include <stdio.h>
int foo() {
#pragma omp target
  ;
  printf("%s\n", "DONE.");
  return 0;
}
#else
#include <dlfcn.h>
#include <stdio.h>
int main(int argc, char **argv) {
  void *Handle = dlopen(argv[1], RTLD_NOW);
  int (*Foo)(void);

  if (Handle == NULL) {
    printf("dlopen() failed: %s\n", dlerror());
    return 1;
  }
  Foo = (int (*)(void)) dlsym(Handle, "foo");
  if (Handle == NULL) {
    printf("dlsym() failed: %s\n", dlerror());
    return 1;
  }
  // CHECK: DONE.
  // CHECK-NOT: {{abort|fault}}
  return Foo();
}
#endif
