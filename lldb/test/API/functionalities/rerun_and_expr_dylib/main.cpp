#include <cassert>
#include <dlfcn.h>

extern struct Foo imported;

int main() {
  void *handle = dlopen("libfoo.dylib", RTLD_NOW);
  struct Foo *foo = (struct Foo *)dlsym(handle, "global_foo");
  assert(foo != nullptr);

  return 0;
}
