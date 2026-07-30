#include "dylib.h"
#include <cassert>
#include <cstdio>
#include <thread>
#include <chrono>

int main(int argc, char* argv[]) {
  // Break here before we dlopen the 'liblib_b.so' shared library.
  void* dylib_handle = dylib_open("lib_b"); 
  assert(dylib_handle && "dlopen failed");
  void (*func_handle)() = (void (*)()) dylib_get_symbol(dylib_handle, "b_function");
  assert(func_handle && "dlsym failed");
  func_handle();
  return 0;
}
