// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak


class A<list<list<int>> vals> {
  list<int> first = vals[0];
  list<int> rest  = !cond(!empty(!tail(vals)): vals[0],
                          1                 : vals[1]);
}

def A_OneEl : A<[[1,2,3]]>;
// CHECK:      def A_OneEl {  // A
// CHECK-NEXT: list<int> first = [1, 2, 3];
// CHECK-NEXT: list<int> rest = [1, 2, 3];
// CHECK-NEXT: }

def A_TwoEl : A<[[1,2,3], [4,5,6]]>;
// CHECK:      def A_TwoEl { // A
// CHECK-NEXT: list<int> first = [1, 2, 3];
// CHECK-NEXT: list<int> rest = [4, 5, 6];
// CHECK-NEXT: }


class B<list<int> v> {
  list<int> vals = v;
}
class BB<list<list<int>> vals> : B<!cond(!empty(!tail(vals)): vals[0],  1 : vals[1])>;
class BBB<list<list<int>> vals> : BB<vals>;

def B_OneEl : BBB<[[1,2,3]]>;
// CHECK:      def B_OneEl { //  B BB BBB
// CHECK-NEXT: list<int> vals = [1, 2, 3];
// CHECK-NEXT: }

def B_TwoEl : BBB<[[1,2,3],[4,5,6]]>;
// CHECK:      def B_TwoEl { // B BB BBB
// CHECK-NEXT: list<int> vals = [4, 5, 6];
// CHECK-NEXT: }
