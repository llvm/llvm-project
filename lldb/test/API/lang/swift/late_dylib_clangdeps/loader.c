#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <libgen.h>

int main(int argc, const char **argv) {
  char *dylib_name = strcat(dirname(argv[0]),"/dylib.dylib");
  void *dylib = dlopen(dylib_name, RTLD_NOW);
  void (*f)() = dlsym(dylib, "f");
  f();
  return 0;
}
