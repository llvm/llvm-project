#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#include <assert.h>
#include <stdio.h>
#include <thread>
#include <chrono>

// We do not use the dylib.h implementation, because
// we need to pass full path to the dylib.
void* dylib_open(const char* full_path) {
#ifdef _WIN32
  return LoadLibraryA(full_path);
#else
  return dlopen(full_path, RTLD_LAZY);
#endif
}

int main(int argc, char* argv[]) {
  assert(argc == 2 && "argv[1] must be the full path to lib_b library");
  const char* dylib_full_path= argv[1];
  printf("Using dylib at: %s\n", dylib_full_path);

  // Wait until debugger is attached.
  int main_thread_continue = 0;
  int i = 0;
  int timeout = 10;
  for (i = 0; i < timeout; i++) {
    std::this_thread::sleep_for(std::chrono::seconds(1));  // break here
    if (main_thread_continue) {
      break;
    }
  }
  assert(i != timeout && "timed out waiting for debugger");

  // dlopen the 'liblib_b.so' shared library.
  void* dylib_handle = dylib_open(dylib_full_path);
  assert(dylib_handle && "dlopen failed");

  return i; // break after dlopen
}
