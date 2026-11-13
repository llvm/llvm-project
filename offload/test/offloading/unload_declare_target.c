// Unloading a library must drop the mappings of its declare-target globals.
// Otherwise reloading it finds the stale mapping (host address -> freed device
// address) and uses it instead of the new program's global.

// clang-format off
// RUN: %libomptarget-compile-generic -DSHARED -fPIC -shared -o %t.so && %libomptarget-compile-generic -o %t -ldl && %libomptarget-run-generic %t.so 2>&1 | %fcheck-generic
// clang-format on

#ifdef SHARED
int Global = 10;
#pragma omp declare target(Global)

// Increment the device copy of Global and return its device value.
int bump(int Add) {
  int Result = -1;
#pragma omp target map(from : Result)
  {
    Global += Add;
    Result = Global;
  }
  return Result;
}

// Copy the device copy of Global back to the host through the mapping table.
int fetch(void) {
#pragma omp target update from(Global)
  return Global;
}

void *addr(void) { return &Global; }
#else
#include <dlfcn.h>
#include <omp.h>
#include <stdio.h>

typedef int (*FnTy)(int);
typedef int (*FetchTy)(void);

static int load(const char *Path, void **Handle, FnTy *Bump, FetchTy *Fetch) {
  *Handle = dlopen(Path, RTLD_NOW);
  if (!*Handle) {
    printf("dlopen() failed: %s\n", dlerror());
    return 1;
  }
  *Bump = (FnTy)dlsym(*Handle, "bump");
  *Fetch = (FetchTy)dlsym(*Handle, "fetch");
  if (!*Bump || !*Fetch) {
    printf("dlsym() failed: %s\n", dlerror());
    return 1;
  }
  return 0;
}

int main(int argc, char **argv) {
  for (int Iter = 0; Iter < 2; ++Iter) {
    void *Handle;
    FnTy Bump;
    FetchTy Fetch;
    if (load(argv[1], &Handle, &Bump, &Fetch))
      return 1;

    // Each load gets a fresh program, so Global starts again at 10.
    int R = Bump(5);
    int F = Fetch();
    printf("iter %d: bump=%d fetch=%d\n", Iter, R, F);

    // The global is mapped while its library is loaded.
    void *(*Addr)(void) = (void *(*)(void))dlsym(Handle, "addr");
    void *GlobalAddr = Addr();
    int Dev = omp_get_default_device();
    printf("iter %d: present=%d\n", Iter,
           omp_target_is_present(GlobalAddr, Dev));

    if (dlclose(Handle)) {
      printf("dlclose() failed: %s\n", dlerror());
      return 1;
    }

    // Unloading the library must have dropped the mapping. Only the address
    // is used, the memory itself is gone.
    printf("iter %d: after unload present=%d\n", Iter,
           omp_target_is_present(GlobalAddr, Dev));
  }
  printf("DONE.\n");
  return 0;
}
// CHECK: iter 0: bump=15 fetch=15
// CHECK: iter 0: present=1
// CHECK: iter 0: after unload present=0
// CHECK: iter 1: bump=15 fetch=15
// CHECK: iter 1: present=1
// CHECK: iter 1: after unload present=0
// CHECK: DONE.
// CHECK-NOT: {{abort|fault}}
#endif
