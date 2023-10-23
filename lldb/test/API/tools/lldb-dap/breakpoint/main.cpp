#include <dlfcn.h>
#include <stdexcept>
#include <stdio.h>

int twelve(int i) {
  return 12 + i; // break 12
}

int thirteen(int i) {
  return 13 + i; // break 13
}

namespace a {
int fourteen(int i) {
  return 14 + i; // break 14
}
} // namespace a
int main(int argc, char const *argv[]) {
#if defined(__APPLE__)
  const char *libother_name = "libother.dylib";
#else
  const char *libother_name = "libother.so";
#endif

  void *handle = dlopen(libother_name, RTLD_NOW);
  if (handle == nullptr) {
    fprintf(stderr, "%s\n", dlerror());
    exit(1);
  }

  int (*foo)(int) = (int (*)(int))dlsym(handle, "foo");
  if (foo == nullptr) {
    fprintf(stderr, "%s\n", dlerror());
    exit(2);
  }
  foo(12); // before loop

  for (int i = 0; i < 10; ++i) {
    int x = twelve(i) + thirteen(i) + a::fourteen(i); // break loop
  }
  try {
    throw std::invalid_argument("throwing exception for testing");
  } catch (...) {
    puts("caught exception...");
  }
  return 0; // after loop
}
