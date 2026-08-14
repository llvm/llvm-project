#include <dlfcn.h>
#include <signal.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <path-to-libtestlib.so>\n", argv[0]);
    return 1;
  }

  void *handle = dlopen(argv[1], RTLD_NOW);
  if (!handle) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }

  int (*func)(int) = dlsym(handle, "lib_func");
  if (!func) {
    fprintf(stderr, "dlsym failed: %s\n", dlerror());
    return 1;
  }

  printf("result: %d\n", func(21));
  raise(SIGSTOP); // stop here for the debugger
  dlclose(handle);
  return 0;
}
