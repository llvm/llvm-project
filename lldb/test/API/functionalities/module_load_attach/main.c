#include <dlfcn.h>
#include <assert.h>
#include <unistd.h>

volatile int flip_to_1_to_continue = 0;

int main() {
  lldb_enable_attach();
  while (! flip_to_1_to_continue) // Wait for debugger to attach
    sleep(1);
  // dlopen the feature
  void *feature = dlopen("libfeature.so", RTLD_NOW);
  assert(feature && "dlopen failed?");
  return 0; // break after dlopen
}
