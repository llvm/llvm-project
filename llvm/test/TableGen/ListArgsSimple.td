// RUN: llvm-tblgen %s
// XFAIL: vg_leak

class B<int v> {
  int val = v;
}

class BB<list<int> vals> : B<vals[0]>;
class BBB<list<int> vals> : BB<vals>;
