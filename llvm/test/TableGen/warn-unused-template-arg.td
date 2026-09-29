// RUN: llvm-tblgen %s 2>&1 | FileCheck %s
// RUN: llvm-tblgen --no-warn-on-unused-template-args %s 2>&1 | FileCheck %s --check-prefix=CHECK-DISABLED

class UnusedClassArg<int foo> {}

// CHECK: warning: unused template argument: UnusedClassArg:foo
// CHECK-NEXT: class UnusedClassArg<int foo> {}
// CHECK-NEXT:                          ^

multiclass UnusedMultiClassArg<int foo> {
  def bar;
}

defm : UnusedMultiClassArg<1>;

// CHECK: warning: unused template argument: UnusedMultiClassArg::foo
// CHECK-NEXT: multiclass UnusedMultiClassArg<int foo> {
// CHECK-NEXT:                                    ^

class NoWarning<int b> {
  int a = b;
}

// CHECK-NOT: warning: unused template argument: NoWarning:b
// CHECK-DISABLED-NOT: warning
