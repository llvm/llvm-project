// clang-format off
// RUN: %libomptarget-compileopt-generic -loffload.kernels -mllvm -enable-gpu-san
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

int main(void) {

  int *Null = 0;
#pragma omp target
  {
    // clang-format off
    // CHECK:      ERROR: AddressSanitizer out-of-bounds-access on address 0x0 at pc [[PC:0x.*]]
    // CHECK-NEXT: WRITE of size 4 at 0x0 thread B<0,0,0> T<0,0,0>
    // CHECK-NEXT: #0 [[PC]] main null.c:[[@LINE+3]]
    // CHECK-NEXT: 0x0 is located 0 bytes inside of 0-byte region [0x0,0x0)
    // clang-format on
    *Null = 42;
  }
}
