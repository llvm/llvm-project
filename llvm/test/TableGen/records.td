// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not llvm-tblgen -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: not llvm-tblgen -DERROR3 %s 2>&1 | FileCheck --check-prefix=ERROR3 %s
// XFAIL: vg_leak

class A;
def a0 : A;
def a1 : A;

class B : A;
def b0 : B;
def b1 : B;

def instances_A {
  list<A> instances = !instances<A>();
}

def instances_A_x0 {
  list<A> instances = !instances<A>(".*0");
}

def instances_A_x1 {
  list<A> instances = !instances<A>(".*1");
}

def instances_B {
  list<B> instances = !instances<B>();
}

// CHECK-LABEL: def instances_A {
// CHECK-NEXT:    list<A> instances = [a0, a1, b0, b1];
// CHECK-NEXT:  }

// CHECK-LABEL: def instances_A_x0 {
// CHECK-NEXT:    list<A> instances = [a0, b0];
// CHECK-NEXT:  }

// CHECK-LABEL: def instances_A_x1 {
// CHECK-NEXT:    list<A> instances = [a1, b1];
// CHECK-NEXT:  }

// CHECK-LABEL: def instances_B {
// CHECK-NEXT:    list<B> instances = [b0, b1];
// CHECK-NEXT:  }

#ifdef ERROR1
defvar error1 = !instances<A>(123)
// ERROR1: error: expected string type argument in !instances operator
#endif

#ifdef ERROR2
defvar error2 = !instances<1>("")
// ERROR2: error: Unknown token when expecting a type
#endif

#ifdef ERROR3
defvar error3 = !instances<A>("([)]")
// ERROR3: error: invalid regex '([)]'
#endif
