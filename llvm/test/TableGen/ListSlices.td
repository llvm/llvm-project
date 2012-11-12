// RUN: llvm-tblgen %s
// XFAIL: vg_leak

def A {
  list<int> B = [10, 20, 30, 4, 1, 1231, 20];
}

def B {
  list<int> X = [10, 20, 30, 4, 1, 1231, 20] [2-4,2,2,0-6];

  list<int> Y = X[4,5];
  int Z = X[4];

  list<int> C = A.B[1-4];

  list<list<int>> AA = [X, Y];

  int BB = AA[0][1];
}
