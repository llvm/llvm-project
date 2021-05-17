// RUN: %libomptarget-compile-generic -DSHARED -fPIC -shared -o %t.so && %clang %flags %s -o %t -ldl && %libomptarget-run-generic %t.so 2>&1 | %fcheck-generic

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
