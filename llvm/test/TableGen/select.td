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

def select_A {
  list<A> selected = !select<A>(".*");
}

def select_A_x0 {
  list<A> selected = !select<A>(".*0");
}

def select_A_x1 {
  list<A> selected = !select<A>(".*1");
}

def select_B {
  list<B> selected = !select<B>(".*");
}

// CHECK-LABEL: def select_A {
// CHECK-NEXT:    list<A> selected = [a0, a1, b0, b1];
// CHECK-NEXT:  }

// CHECK-LABEL: def select_A_x0 {
// CHECK-NEXT:    list<A> selected = [a0, b0];
// CHECK-NEXT:  }

// CHECK-LABEL: def select_A_x1 {
// CHECK-NEXT:    list<A> selected = [a1, b1];
// CHECK-NEXT:  }

// CHECK-LABEL: def select_B {
// CHECK-NEXT:    list<B> selected = [b0, b1];
// CHECK-NEXT:  }

#ifdef ERROR1
defvar error1 = !select<A>(123)
// ERROR1: error: expected string type argument in !select operator
#endif

#ifdef ERROR2
defvar error2 = !select<1>("")
// ERROR2: error: Unknown token when expecting a type
#endif

#ifdef ERROR3
defvar error3 = !select<A>("([)]")
// ERROR3: error: invalid regex '([)]'
#endif
