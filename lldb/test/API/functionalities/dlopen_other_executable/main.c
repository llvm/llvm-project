#include <dlfcn.h>
#include <assert.h>

int main() {
  int i = 0; // break here
  // dlopen the 'other' test executable.
  int h = dlopen("other", RTLD_LAZY);
  assert(h && "dlopen failed?");
  return i; // break after dlopen
}
