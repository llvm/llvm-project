// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: --- Defs ---

// CHECK: def A1 {
// CHECK:   int ret = 0;
// CHECK: }

// CHECK: def A2 {
// CHECK:   int ret = 5;
// CHECK: }

// CHECK: def A3 {
// CHECK:   int ret = 10;
// CHECK: }

// CHECK: def B1 {
// CHECK:   list<string> ret = [];
// CHECK: }

// CHECK: def B2 {
// CHECK:   list<string> ret = [];
// CHECK: }

// CHECK: def B3 {
// CHECK:   list<string> ret = ["a"];
// CHECK: }

// CHECK: def B4 {
// CHECK:   list<string> ret = ["a", "b", "c", "d"];
// CHECK: }

// CHECK: def E0 {
// CHECK:   list<int> ret = [45, 45, 45, 45];
// CHECK: }

class Sum<list<int> lst> {
  int ret = !foldl(0, lst, a, b, !add(a, b));
}

class Flatten<list<list<string>> lst> {
  list<string> ret = !foldl([]<string>, lst, a, b, !listconcat(a, b));
}

def A1 : Sum<[]>;
def A2 : Sum<[5]>;
def A3 : Sum<[1, 2, 3, 4]>;

def B1 : Flatten<[]>;
def B2 : Flatten<[[]]>;
def B3 : Flatten<[["a"]]>;
def B4 : Flatten<[["a", "b"], [], ["c"], ["d"]]>;

// The variables a and b are declared both in the "inner" foldl and in the
// other foreach. The test checks that they don't "leak".
class C<list<int> lst> {
  int ret = !foldl(0, lst, a, b, !add(a, b));
}

class D<list<int> lst1, list<int> lst2> {
  list<int> x = !foreach(a, lst1, C<lst2>.ret);
  list<int> y = !foreach(b, lst1, C<lst2>.ret);
  list<int> z = !listconcat(x, y);
}

class E<list<int> lst2> {
  list<int> ret = D<[0, 1], lst2>.z;
}

def E0 : E<[10, 15, 20]>;
