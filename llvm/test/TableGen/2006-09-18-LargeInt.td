// RUN: llvm-tblgen %s | grep -- 4294901760
// XFAIL: vg_leak

def X {
  int Y = 0xFFFF0000;
}
