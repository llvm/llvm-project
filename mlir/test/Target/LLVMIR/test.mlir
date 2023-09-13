// RUN: mlir-translate -test-to-llvmir -split-input-file %s | FileCheck %s

module {
  "test.symbol"() <{sym_name = "foo"}> : () -> ()
}

// CHECK-NOT: @sym_from_attr
// CHECK: @foo = external global i32
// CHECK-NOT: @sym_from_attr

// -----

// Make sure that the module attribute is processed before its body, so that the
// `test.symbol` that is created as a result of the `test.discardable_mod_attr`
// attribute is later picked up and translated to LLVM IR.
module attributes {test.discardable_mod_attr = true} {}

// CHECK: @sym_from_attr = external global i32
