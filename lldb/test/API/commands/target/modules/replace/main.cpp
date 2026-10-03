#include <cstdio>
#include <dlfcn.h>

int main(int argc, char **argv) {
  if (argc < 2)
    return 1;

  void *handle = dlopen(argv[1], RTLD_NOW);
  if (!handle)
    return 2;

  int (*common_func)() = (int (*)())dlsym(handle, "common_func");
  int (*get_tls_var)() = (int (*)())dlsym(handle, "get_tls_var");
  if (!common_func || !get_tls_var)
    return 3;

  // Call through to the library's thread-local before stopping, so that its
  // TLS block has actually been allocated for this thread by the time the
  // test looks at it.
  int tls = get_tls_var();

  printf("%d %d\n", common_func(), tls); // break after dlopen
  return 0;
}
