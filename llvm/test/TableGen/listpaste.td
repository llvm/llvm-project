// RUN: llvm-tblgen %s | FileCheck %s

// CHECK: class A<bit A:x = ?> {
// CHECK: list<int> lst = !listconcat([], !if(A:x, [], [4]));
// CHECK: }
class A<bit x> {
  list<int> lst = []  # !if(x, [], [4]);
}

// CHECK: class A1<list<int> A1:l = ?> {
// CHECK:   list<int> A1List = A1:l;
// CHECK: }
class A1<list<int> l> {
  list<int> A1List = l;
}

// CHECK: def A0 {
// CHECK:   list<int> lst = [4];
// CHECK: }
def A0 : A<0>;

// CHECK: def A1 {
// CHECK:   list<int> lst = [];
// CHECK: }
def A1 : A<1>;

// CHECK: def A1_0 {
// CHECK:   list<int> A1List = [1, 2, 3, 4];
// CHECK: }
def A1_0 : A1<[1,2] # [3,4]>;

// CHECK: def A1_1 {
// CHECK:   list<int> A1List = [1, 2];
// CHECK: }
def A1_1 : A1<[] # [1,2]>;

// CHECK: def A1_2 {      // A1
// CHECK:   list<int> A1List = [1, 2];
// CHECK: }
def A1_2 : A1<[1,2] # []>;
