#include "service.h"
#include <cassert>
#include <dlfcn.h>

#ifndef PLUGIN_PATH
#error "Expected PLUGIN_PATH to be defined"
#endif // !PLUGIN_PATH

int main() {
  void *handle = dlopen(PLUGIN_PATH, RTLD_NOW);
  assert(handle != nullptr);
  void (*plugin_init)(void) = (void (*)(void))dlsym(handle, "plugin_init");
  assert(plugin_init != nullptr);
  void (*plugin_entry)(void) = (void (*)(void))dlsym(handle, "plugin_entry");
  assert(plugin_entry != nullptr);

  exported();
  plugin_init();
  plugin_entry();
  return 0;
}
