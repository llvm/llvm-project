#include <ptrcheck.h>
#include <stdlib.h>

#ifdef _WIN32
#include <Windows.h>
#define DLOPEN(name) LoadLibraryA(name)
#define DLSYM(handle, name) GetProcAddress((HMODULE)handle, name)
#define DLCLOSE(handle) FreeLibrary((HMODULE)handle)
#define DYLIB_NAME "dylib_with_soft_trap.dll"
#else
#include <dlfcn.h>
#define DLOPEN(name) dlopen(name, RTLD_NOW)
#define DLSYM(handle, name) dlsym(handle, name)
#define DLCLOSE(handle) dlclose(handle)
#ifdef __APPLE__
#define DYLIB_NAME "libdylib_with_soft_trap.dylib"
#else
#define DYLIB_NAME "libdylib_with_soft_trap.so"
#endif
#endif

#ifdef _WIN32
// Windows searches the executable's directory for DLLs.
#define DYLIB_PATH DYLIB_NAME
#else
#define DYLIB_PATH DYLIB_DIR "/" DYLIB_NAME
#endif

__attribute__((noinline, optnone)) void test_breakpoint(void) {}
__attribute__((noinline, optnone)) void test_breakpoint_2(void) {}
__attribute__((noinline, optnone)) void test_breakpoint_dlopen(void) {}
__attribute__((noinline, optnone)) void test_breakpoint_after_dlopen(void) {}

int main(void) {
  int pad;
  int buffer[] = {0, 1};
  int pad2;
  int tmp = buffer[2]; // first soft trap: access past upper bound
  test_breakpoint();
  tmp = buffer[-1]; // second soft trap: access below lower bound

  if (getenv("DO_EXTRA_SOFT_TRAPS")) {
    tmp = buffer[3]; // third soft trap: access past upper bound
    test_breakpoint_2();
    tmp = buffer[-2]; // fourth soft trap: access below lower bound
  }

  if (getenv("DO_DLOPEN")) {
    // dlopen a shared library that contains the soft trap runtime functions.
    // This triggers ModulesDidLoad which could re-activate a disabled plugin.
    test_breakpoint_dlopen();
    void *__unsafe_indexable handle = DLOPEN(DYLIB_PATH);
    if (handle) {
      typedef int (*dylib_func_t)(void);
      dylib_func_t func =
          __unsafe_forge_single(dylib_func_t, DLSYM(handle, "dylib_soft_trap"));
      if (func)
        tmp += func(); // soft trap inside dylib
      test_breakpoint_after_dlopen();
      DLCLOSE(handle);
    }
  }

  return 0;
}
