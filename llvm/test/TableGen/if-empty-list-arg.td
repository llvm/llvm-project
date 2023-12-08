// RUN: llvm-tblgen %s
// XFAIL: vg_leak

class C<bit cond> {
  list<int> X = !if(cond, [1, 2, 3], []);
  list<int> Y = !if(cond, [], [4, 5, 6]);
}
