// RUN: llvm-tblgen %s
// XFAIL: vg_leak

class C<bit cond> {
  bit true = 1;
  list<int> X = !cond(cond: [1, 2, 3], true : []);
  list<int> Y = !cond(cond: [], true : [4, 5, 6]);
}
