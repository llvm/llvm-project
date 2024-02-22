#include <dlfcn.h>
#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

int main(int argc, const char **argv) {
  char dylib_name[PATH_MAX];
  strlcpy(dylib_name, dirname(argv[0]), PATH_MAX);
  strlcat(dylib_name, "/dylib.dylib", PATH_MAX);
  void *dylib = dlopen(dylib_name, RTLD_NOW);
  void (*f)() = dlsym(dylib, "f");
  f();
  return 0;
}
