// RUN: llvm-tblgen %s | FileCheck %s

// Check the order and values of the fields of a def from a multiclass when the
// defm adds more superclasses. A def lists the fields it inherits from its
// superclasses before the fields declared in its body, so new fields from the
// superclasses added by the defm come before the fields from the def's body,
// even though those superclasses were added later. The order of fields is not
// semantically meaningful.

class Base {
  int A = 1;
  int B = 2;
}

class Extra {
  int A = 10;
  int E = 3;
}

class Extra2 {
  int E = 30;
  int F = 4;
}

multiclass M {
  def NAME : Base {
    int C = 5;
  }
}

// CHECK-LABEL: def X {
// CHECK-NEXT:    int A = 10;
// CHECK-NEXT:    int B = 2;
// CHECK-NEXT:    int E = 3;
// CHECK-NEXT:    int C = 5;
// CHECK-NEXT:  }
defm X : M, Extra;

// CHECK-LABEL: def Y {
// CHECK-NEXT:    int A = 10;
// CHECK-NEXT:    int B = 2;
// CHECK-NEXT:    int E = 30;
// CHECK-NEXT:    int F = 4;
// CHECK-NEXT:    int C = 5;
// CHECK-NEXT:  }
defm Y : M, Extra, Extra2;
