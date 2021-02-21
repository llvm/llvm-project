#include <dlfcn.h>
#include <assert.h>
#include <unistd.h>

volatile int flip_to_1_to_continue = 0;

int main() {
  lldb_enable_attach();
  while (! flip_to_1_to_continue) // Wait for debugger to attach
    sleep(1);
  void *dylib = dlopen("libdylib.so", RTLD_LAZY);
  assert(dylib && "dlopen failed?");
  return 0; // break after dlopen
}
