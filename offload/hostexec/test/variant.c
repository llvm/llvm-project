#include <hostexec.h>
#include <omp.h>
#include <stdarg.h>
#include <stdio.h>
// #include <unistd.h>

// In this test case, think of my_increment as some library function that may
// only execute on the host. When my_increment is called from a target region,
// the device variant of my_increment will use hostexec_int to run my_increment
// on the host.

extern int my_increment(int val) {
  int rval = val + 1;
  return rval;
}

//========== START of example supplemental library header ==========
//
//  To enable host-only execution inside an OpenMP target region of any
//  library function without source modification in the target region,
//  the library maintainer would create a "supplemental" header.
//  This supplemental header has 4 parts for each host-only library function
//  This is alternative to source modifications to use reverse offload.
//  This supplemental header could be auto-generated from information found
//  in the library function declaration.

// 1. Create variadic proxy function for each host-only library function.
//    This variadic proxy function builds a non-variadic call site on the
//    host which results in a correct stack frame. The host service thread
//    calls the variadic function via function pointer passed by hostexec.
//    First, the args are unpacked suitable for a variadic function call.
//    The service thread cannot call a non-variadic library function directly
//    because we cannot recreate every possible function template.
//    Having the library owner be responsible for the conversion from variadic
//    to nonvariadic function provides the ability to use hostexec for
//    any library function without changing the user code.
extern int V_my_increment(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  int v0 = va_arg(args, int);
  va_end(args);
  return my_increment(v0); // The non-variadic call site
}

#pragma omp begin declare target

// 2. Create a global variable to store pointer to variadic proxy function
hostexec_int_t *V_my_increment_var;

// 3. Create the device variant of the library function that calls hostexec
//    with the pointer to the variadic proxy function.
#pragma omp begin declare variant match(device = {arch(amdgcn, nvptx, nvptx64)})
int my_increment(int val) { return hostexec_int(V_my_increment_var, val); }
#pragma omp end declare variant
#pragma omp end declare target

// 4.  Initialize pointers to host-only library functions on the device.
//     These are host pointers stored as device globals passed to hostexec.
void _mylib_set_device_globals() {
  V_my_increment_var = V_my_increment;
#pragma omp target update to(V_my_increment_var)
}

//========== END example supplemental library header ==========

void vmul(int *a, int *b, int *c, int N) {
#pragma omp target teams distribute parallel for map(to : a[0 : N], b[0 : N])  \
    map(from : c[0 : N])
  for (int i = 0; i < N; i++) {
    c[i] = a[i] * b[i];
    c[i] = my_increment(c[i]); // example call to library function.
    c[i]--; // adjust (decrement) to get the correct answer in this test case.
  }
}

int main() {
  // Initialize pointers to host-only library functions on device.
  // FIXME: Get this done by initialization of libomptarget globals.
  _mylib_set_device_globals();

  const int N = 10;
  int a[N], b[N], c[N], validate[N];
  int flag = -1; // Mark Success
  for (int i = 0; i < N; i++) {
    a[i] = i + 1;
    b[i] = i + 2;
    validate[i] = a[i] * b[i];
  }

  vmul(a, b, c, N);

  for (int i = 0; i < N; i++) {
    if (c[i] != validate[i]) {
      //          print 1st bad index
      if (flag == -1)
        printf("First fail: c[%d](%d) != validate[%d](%d)\n", i, c[i], i,
               validate[i]);
      flag = i;
    }
  }
  if (flag == -1) {
    printf("Success\n");
    return 0;
  } else {
    printf("Last fail: c[%d](%d) != validate[%d](%d)\n", flag, c[flag], flag,
           validate[flag]);
    printf("Fail\n");
    return 1;
  }
}
