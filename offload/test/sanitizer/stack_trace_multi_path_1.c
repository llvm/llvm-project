// clang-format off
// : %libomptarget-compileopt-generic -fsanitize=offload -O1
// : not %libomptarget-run-generic 2> %t.out
// : %fcheck-generic --check-prefixes=CHECK < %t.out
// : %libomptarget-compileopt-generic -fsanitize=offload -O3
// : not %libomptarget-run-generic 2> %t.out
// RUN: %libomptarget-compileopt-generic -fsanitize=offload -O3 -g
// RUN: not %libomptarget-run-generic 2> %t.out
// RUN: %fcheck-generic --check-prefixes=DEBUG < %t.out
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

[[clang::optnone]] int deref(int *P) { return *P; }

[[gnu::noinline]] int bar(int *P) { return deref(P); }
[[gnu::noinline]] int baz(int *P) { return deref(P); }

int main(void) {

#pragma omp target
  {
    int *NullPtr = 0;
    int X;
    int *Valid = &X;
    // clang-format off
    // CHECK: ERROR: OffloadSanitizer out-of-bounds access on address 0x0000000000000000 at pc [[PC:.*]]
    // CHECK: WRITE of size 4 at 0x0000000000000000 thread <0, 0, 0> block <0, 0, 0> (acc 1, heap)
    // CHECK:     #0 [[PC]] omp target (main:[[@LINE-6]]) in <unknown>:0
    // 
    // CHECK: 0x0000000000000000 is located 0 bytes inside of a 0-byte region [0x0000000000000000,0x0000000000000000)
    //
    // DEBUG: ERROR: OffloadSanitizer out-of-bounds access on address 0x0000000000000000 at pc [[PC:.*]]
    // DEBUG: WRITE of size 4 at 0x0000000000000000 thread <0, 0, 0> block <0, 0, 0> (acc 1, heap)
    // DEBUG:     #0 [[PC]] omp target (main:[[@LINE-12]]) in {{.*}}volatile_stack_null.c:[[@LINE+4]]
    // 
    // DEBUG: 0x0000000000000000 is located 0 bytes inside of a 0-byte region [0x0000000000000000,0x0000000000000000)
    // clang-format on
    bar(Valid);
    baz(NullPtr);
  }
}
