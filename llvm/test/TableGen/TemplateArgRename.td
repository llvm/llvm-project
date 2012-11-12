// RUN: llvm-tblgen %s
// XFAIL: vg_leak

// Make sure there is no collision between XX and XX.
def S;

class Before<int XX>;
class After : Before<4> {
  dag XX = (S);
}



class C1<int X> {
  int Y = X;
}
class C2<int Y, dag X> : C1<Y>;

