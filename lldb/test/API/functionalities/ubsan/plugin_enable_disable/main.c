#include <stdint.h>
#include <stdlib.h>

#ifdef _WIN32
#include <Windows.h>
#define DLOPEN(name) LoadLibraryA(name)
#define DLSYM(handle, name) GetProcAddress((HMODULE)handle, name)
#define DLCLOSE(handle) FreeLibrary((HMODULE)handle)
#define DYLIB_NAME "dylib_with_ubsan_issue.dll"
#else
#include <dlfcn.h>
#define DLOPEN(name) dlopen(name, RTLD_NOW)
#define DLSYM(handle, name) dlsym(handle, name)
#define DLCLOSE(handle) dlclose(handle)
#ifdef __APPLE__
#define DYLIB_NAME "libdylib_with_ubsan_issue.dylib"
#else
#define DYLIB_NAME "libdylib_with_ubsan_issue.so"
#endif
#endif

__attribute__((noinline, optnone)) void test_breakpoint(void) {}
__attribute__((noinline, optnone)) void test_breakpoint_2(void) {}
__attribute__((noinline, optnone)) void test_breakpoint_dlopen(void) {}

__attribute__((noinline, optnone)) int shift(void) { return 33; }

int main() {
  uint32_t x = 0;

  x = x << shift(); // first ubsan issue
  x = x << shift(); // second ubsan issue
  test_breakpoint();
  x = x << shift(); // third ubsan issue
  test_breakpoint_2();
  x = x << shift(); // fourth ubsan issue

  if (getenv("DO_DLOPEN")) {
    // dlopen a shared library. This triggers ModulesDidLoad which could
    // re-activate a disabled plugin.
    test_breakpoint_dlopen();
    void *handle = DLOPEN(DYLIB_NAME);
    if (handle) {
      int (*func)(void) = (int (*)(void))DLSYM(handle, "dylib_ubsan_issue");
      if (func)
        x += func(); // ubsan issue inside dylib
      DLCLOSE(handle);
    }
  }

  return 0;
}
