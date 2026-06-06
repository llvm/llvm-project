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

// -----

// CHECK-LABEL: @dialect_attr_translation
llvm.func @dialect_attr_translation() {
  // CHECK: ret void, !annotation ![[MD_ID:.+]]
  llvm.return {test.add_annotation}
}
// CHECK: ![[MD_ID]] = !{!"annotation_from_test"}

// -----

// CHECK-LABEL: @dialect_attr_translation_multi
llvm.func @dialect_attr_translation_multi(%a: i64, %b: i64, %c: i64) -> i64 {
  // CHECK: add {{.*}}, !annotation ![[MD_ID_ADD:.+]]
  // CHECK: mul {{.*}}, !annotation ![[MD_ID_MUL:.+]]
  // CHECK: ret {{.*}}, !annotation ![[MD_ID_RET:.+]]
  %ab = llvm.add %a, %b {test.add_annotation = "add"} : i64
  %r = llvm.mul %ab, %c {test.add_annotation = "mul"} : i64
  llvm.return {test.add_annotation = "ret"} %r : i64
}
// CHECK-DAG: ![[MD_ID_ADD]] = !{!"annotation_from_test: add"}
// CHECK-DAG: ![[MD_ID_MUL]] = !{!"annotation_from_test: mul"}
// CHECK-DAG: ![[MD_ID_RET]] = !{!"annotation_from_test: ret"}
