// clang-format off
// RUN: %libomptarget-compilexx-generic && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 | %fcheck-generic
// clang-format on

// REQUIRES: libomptarget-debug

struct DataTy {
  float a;
  float b[3];
};

int main(int argc, char **argv) {
  DataTy D;
#pragma omp target map(D.a) map(D.b[ : 2])
  {
    D.a = 0;
    D.b[0] = 1;
  }
  return 0;
}
// clang-format off
// CHECK: omptarget --> Entry  0: Base=[[DAT_HST_PTR_BASE:0x.*]], Begin=[[DAT_HST_PTR_BASE]], Size=12
// CHECK: omptarget --> Entry  1: Base=[[DAT_HST_PTR_BASE]], Begin=[[DAT_HST_PTR_BASE]], Size=4,
// CHECK: omptarget --> Entry  2: Base=[[DAT_HST_PTR_BASE]], Begin=[[DATUM_HST_PTR_BASE:0x.*]], Size=8,
// clang-format on
