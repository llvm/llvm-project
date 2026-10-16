extern int unrelated(void);
#ifdef CALL_MAIN_RUNTIME
extern void call_main_trap(void);
#endif
#ifdef CALL_LIBRARY_RUNTIME
extern void call_library_trap(void);
#endif
#ifdef LATE_LOAD
#include <assert.h>
#include <dlfcn.h>
#endif

int main(int argc, char **argv) {
  int result = unrelated();
#ifdef CALL_MAIN_RUNTIME
  call_main_trap();
#endif
#ifdef CALL_LIBRARY_RUNTIME
  call_library_trap();
#endif
#ifdef LATE_LOAD
  void *library = dlopen(argv[1], RTLD_NOW);
  assert(library);
  void (*call_late_trap)(void) =
      (void (*)(void))dlsym(library, "call_late_trap");
  assert(call_late_trap);
  call_late_trap();
#endif
  return result;
}
