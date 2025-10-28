#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char const *argv[]) {

#if defined(__APPLE__)
  const char *libother_name = "libother.dylib";
#else
  const char *libother_name = "libother.so";
#endif

  printf("before dlopen\n"); // breakpoint 1
  void *handle = dlopen(libother_name, RTLD_NOW);
  int (*foo)(int) = (int (*)(int))dlsym(handle, "foo");
  foo(12);

  printf("before dlclose\n"); // breakpoint 2
  dlclose(handle);
  printf("after dlclose\n"); // breakpoint 3

  return 0; // breakpoint 1
}
