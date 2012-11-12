// RUN: llvm-tblgen %s
// XFAIL: vg_leak
def {
  bit A = 1;
  int B = A;
}
