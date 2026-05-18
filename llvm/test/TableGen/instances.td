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

// CHECK-LABEL: def test0_instances_A {
// CHECK-NEXT:    list<A> instances = [a0, a1, b0, b1];
// CHECK-NEXT:  }
def test0_instances_A {
  list<A> instances = !instances<A>();
}

// CHECK-LABEL: def test1_instances_A_x0 {
// CHECK-NEXT:    list<A> instances = [a0, b0];
// CHECK-NEXT:  }
def test1_instances_A_x0 {
  list<A> instances = !instances<A>(".*0");
}

// CHECK-LABEL: def test2_instances_A_x1 {
// CHECK-NEXT:    list<A> instances = [a1, b1];
// CHECK-NEXT:  }
def test2_instances_A_x1 {
  list<A> instances = !instances<A>(".*1");
}

// CHECK-LABEL: def test3_instances_B {
// CHECK-NEXT:    list<B> instances = [b0, b1];
// CHECK-NEXT:  }
def test3_instances_B {
  list<B> instances = !instances<B>();
}

//-----------------------------------------------------------------------------//

def a2 : A;
def b2 : B;

class ClassTest {
   list<A> instances_A = !instances<A>();
   list<B> instances_B = !instances<B>();
}

def a3 : A;
def b3 : B;

def test4_in_class_def : ClassTest;
// CHECK-LABEL: def test4_in_class_def {
// CHECK-NEXT:    list<A> instances_A = [a0, a1, a2, a3, b0, b1, b2, b3];
// CHECK-NEXT:    list<B> instances_B = [b0, b1, b2, b3];
// CHECK-NEXT:  }

//-----------------------------------------------------------------------------//
// Self-recurrence is not supported, so it won't be count in.

// CHECK-LABEL: def test5_self_recurrence {
// CHECK-NEXT:    list<A> instances_A = [a0, a1, a2, a3, b0, b1, b2, b3];
// CHECK-NEXT:  }
def test5_self_recurrence : A {
   list<A> instances_A = !instances<A>();
}

//-----------------------------------------------------------------------------//
// Test these in multiclasses/loops.

class C {
  list<C> instances_C = !instances<C>();
}

multiclass MultiClassTest {
  foreach i = 0-2 in {
    def "c"#i : C;
  }
}

// CHECK-LABEL: def test6_in_multiclass_def_c0 {
// CHECK-NEXT:    list<C> instances_C = [];
// CHECK-NEXT:  }
// CHECK-LABEL: def test6_in_multiclass_def_c1 {
// CHECK-NEXT:    list<C> instances_C = [test6_in_multiclass_def_c0];
// CHECK-NEXT:  }
// CHECK-LABEL: def test6_in_multiclass_def_c2 {
// CHECK-NEXT:    list<C> instances_C = [test6_in_multiclass_def_c0, test6_in_multiclass_def_c1];
// CHECK-NEXT:  }
defm test6_in_multiclass_def_ : MultiClassTest;

//-----------------------------------------------------------------------------//
// Default argument/temporary actual parameter will be considered as well.
class D<int n>;

class TestArgument<D d = D<0>> {
  list<D> instances_D = !instances<D>();
}

// CHECK-LABEL: def test7_default_arg {
// CHECK-NEXT:    list<D> instances_D = [anonymous_0];
// CHECK-NEXT:  }
def test7_default_arg : TestArgument;

// CHECK-LABEL: def test8_anonymous0_arg {
// CHECK-NEXT:    list<D> instances_D = [anonymous_0, anonymous_1];
// CHECK-NEXT:  }
// CHECK-LABEL: def test8_anonymous1_arg {
// CHECK-NEXT:    list<D> instances_D = [anonymous_0, anonymous_1, anonymous_2];
// CHECK-NEXT:  }
def test8_anonymous0_arg : TestArgument<D<1>>;
def test8_anonymous1_arg : TestArgument<D<2>>;

//-----------------------------------------------------------------------------//

#ifdef ERROR1
defvar error1 = !instances<A>(123);
// ERROR1: error: expected string type argument in !instances operator
#endif

#ifdef ERROR2
defvar error2 = !instances<1>("");
// ERROR2: error: Unknown token when expecting a type
#endif

#ifdef ERROR3
defvar error3 = !instances<A>("([)]");
// ERROR3: error: invalid regex '([)]'
#endif
