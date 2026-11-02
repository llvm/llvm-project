// RUN: llvm-tblgen %s | FileCheck %s

// Check that a template argument is not in scope in its own default value, so
// the default value can refer to a def with the same name, and that earlier
// template arguments are in scope in later default values.

class Op;
def store : Op;

class StoreU<Op store = store> {
  Op Store = store;
}

multiclass StoreM<Op store = store> {
  def NAME {
    Op Store = store;
  }
}

class Two<int a, int b = !add(a, 1)> {
  int B = b;
}

multiclass TwoM<int a, int b = !add(a, 1)> {
  def NAME {
    int B = b;
  }
}

// CHECK-LABEL: def FromClass {
// CHECK-NEXT:    Op Store = store;
def FromClass : StoreU;

// CHECK-LABEL: def FromMultiClass {
// CHECK-NEXT:    Op Store = store;
defm FromMultiClass : StoreM;

// CHECK-LABEL: def TwoFromClass {
// CHECK-NEXT:    int B = 6;
def TwoFromClass : Two<5>;

// CHECK-LABEL: def TwoFromMultiClass {
// CHECK-NEXT:    int B = 8;
defm TwoFromMultiClass : TwoM<7>;
