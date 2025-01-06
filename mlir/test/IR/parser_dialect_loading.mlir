// RUN: mlir-opt -allow-unregistered-dialect --split-input-file %s | FileCheck %s

// This is a testing that a non-qualified attribute in a custom format
// correctly preload the dialect before creating the attribute.
#attr = #test.nested_polynomial<poly=<1 + x**2>>
// CHECK-LABEL: @parse_correctly
llvm.func @parse_correctly() {
  test.containing_int_polynomial_attr #attr
  llvm.return
}

// -----

#attr2 = #test.nested_polynomial2<poly=<1 + x**2>>
// CHECK-LABEL: @parse_correctly_2
llvm.func @parse_correctly_2() {
  test.containing_int_polynomial_attr2 #attr2
  llvm.return
}
