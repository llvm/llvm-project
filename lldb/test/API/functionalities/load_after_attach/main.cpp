#include "dylib.h"
#include <cassert>
#include <cstdio>
#include <thread>
#include <chrono>
#include <fstream>

int main(int argc, char* argv[]) {
  lldb_enable_attach();
  std::ofstream(argv[1]).close();

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
  void* dylib_handle = dylib_open("lib_b");
  assert(dylib_handle && "dlopen failed");

  return i; // break after dlopen
}
