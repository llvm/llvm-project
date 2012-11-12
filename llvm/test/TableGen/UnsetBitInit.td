// RUN: llvm-tblgen %s
// XFAIL: vg_leak
class x {
  field bits<32> A;
}

class y<bits<2> B> : x {
  let A{21-20} = B;
}

def z : y<{0,?}>;
