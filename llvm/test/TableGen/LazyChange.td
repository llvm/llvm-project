// RUN: llvm-tblgen %s | grep "int Y = 3"
// XFAIL: vg_leak

class C {
  int X = 4;
  int Y = X;
}

let X = 3 in
def D : C;    // Y should be 3 too!

