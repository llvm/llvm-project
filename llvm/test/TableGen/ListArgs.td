// RUN: llvm-tblgen %s
// XFAIL: vg_leak

class B<list<int> v> {
  list<int> vals = v;
}

class BB<list<list<int>> vals> : B<vals[0]>;
class BBB<list<list<int>> vals> : BB<vals>;

def OneB : BBB<[[1,2,3]]>;
def TwoB : BBB<[[1,2,3],[4,5,6]]>;
