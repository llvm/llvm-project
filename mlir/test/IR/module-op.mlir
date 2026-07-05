// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -mlir-print-debuginfo -mlir-print-local-scope  | FileCheck %s

// CHECK: module {
module {
}

// -----

// CHECK: module attributes {foo.attr = true} {
module attributes {foo.attr = true} {
}

// -----

// CHECK: module {
module {
  // CHECK-NEXT: "foo.result_op"() : () -> i32
  %result = "foo.result_op"() : () -> i32
}

// -----

// Check that a top-level module is always created, with location info.
// CHECK: module {
// CHECK-NEXT: } loc({{.*}}module-op.mlir{{.*}})

// -----

// Check that the top-level module can be defined via a single module operation.
// CHECK: module {
// CHECK-NOT: module {
module {
}

// -----

// Check that the implicit top-level module is also a name scope for SSA
// values.  This should not crash.
// CHECK: module {
// CHECK: %{{.*}} = "op"
// CHECK: }
%0 = "op"() : () -> i32

// -----

// CHECK-LABEL: module @foo
// CHECK-NOT: attributes
module @foo {
  // CHECK: module
  module {
    // CHECK: module @bar attributes
    module @bar attributes {foo.bar} {
    }
  }
}

// -----

// expected-error@below {{expects at most one data layout attribute}}
// expected-note@below {{'test.another_attribute' is a data layout attribute}}
// expected-note@below {{'test.random_attribute' is a data layout attribute}}
module attributes { test.random_attribute = #dlti.dl_spec<>,
                    test.another_attribute = #dlti.dl_spec<>} {
}
