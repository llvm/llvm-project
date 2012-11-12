// RUN: llvm-tblgen %s
// XFAIL: vg_leak
//
// Test to make sure that lists work with any data-type

class foo {
  list<int> Test = [1, 2, 3];
  list<string> Test2 = ["abc", "xyz", "gtq"];
}
