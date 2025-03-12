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

def defined_A {
  list<A> defined = !defined<A>();
}

def defined_A_x0 {
  list<A> defined = !defined<A>(".*0");
}

def defined_A_x1 {
  list<A> defined = !defined<A>(".*1");
}

def defined_B {
  list<B> defined = !defined<B>();
}

// CHECK-LABEL: def defined_A {
// CHECK-NEXT:    list<A> defined = [a0, a1, b0, b1];
// CHECK-NEXT:  }

// CHECK-LABEL: def defined_A_x0 {
// CHECK-NEXT:    list<A> defined = [a0, b0];
// CHECK-NEXT:  }

// CHECK-LABEL: def defined_A_x1 {
// CHECK-NEXT:    list<A> defined = [a1, b1];
// CHECK-NEXT:  }

// CHECK-LABEL: def defined_B {
// CHECK-NEXT:    list<B> defined = [b0, b1];
// CHECK-NEXT:  }

#ifdef ERROR1
defvar error1 = !defined<A>(123)
// ERROR1: error: expected string type argument in !defined operator
#endif

#ifdef ERROR2
defvar error2 = !defined<1>("")
// ERROR2: error: Unknown token when expecting a type
#endif

#ifdef ERROR3
defvar error3 = !defined<A>("([)]")
// ERROR3: error: invalid regex '([)]'
#endif
