// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: def range_op_ranges {
def range_op_ranges {
  // !range(n) generates half-open interval "[0,n)"
  // CHECK: list<int> idxs8 = [0, 1, 2, 3, 4, 5, 6, 7];
  list<int> idxs8 = !range(8);

  // !range(m, n) generates half-open interval "[m,n)"
  // CHECK: list<int> idxs4 = [4, 5, 6, 7];
  list<int> idxs4 = !range(4, 8);

  // !range(m, n, s) generates half-open interval "[m,n)" with step s
  // CHECK: list<int> step = [0, 2, 4, 6];
  list<int> step = !range(0, 8, 2);

  // Step can be negative.
  // CHECK: list<int> negative_step = [8, 6, 4, 2];
  list<int> negative_step = !range(8, 0, -2);

  // !range(m, n) generates empty set if m >= n
  // CHECK: list<int> idxs84 = [];
  list<int> idxs84 = !range(8, 4);

  // !range(m, n, s) generates empty set if m < n and s < 0
  // CHECK: list<int> emptycase0 = [];
  list<int> emptycase0 = !range(4, 8, -1);

  // !range(m, n, s) generates empty set if m > n and s > 0
  // CHECK: list<int> emptycase1 = [];
  list<int> emptycase1 = !range(8, 4, 1);

  // !range(list) generates index values for the list. (Not a copy of the list)
  // CHECK: list<int> idxsl = [0, 1, 2, 3];
  list<int> idxsl = !range(idxs4);

  // !range(emptylist) generates empty set
  // CHECK: list<int> idxs0 = [];
  list<int> idxs0 = !range(idxs84);

  // Example: Dynamic !range(n)
  // CHECK: list<list<int>> rn = {{\[}}[0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6]];
  list<list<int>> rn = !foreach(i, idxs4, !range(i));

  // Example: Dynamic !range(m, n)
  // CHECK: list<list<int>> rm = {{\[}}[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5], [2, 3, 4], [3], [], [], [], []];
  list<list<int>> rm = !foreach(i, idxs8, !range(i, !sub(7, i)));

  // Example: Dynamic !range(list)
  // CHECK: list<list<int>> rl = {{\[}}[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4], [0, 1, 2], [0], [], [], [], []];
  list<list<int>> rl = !foreach(r, rm, !range(r));
}
