// RUN: mlir-opt %s -test-convert-call-op | FileCheck %s

// CHECK-LABEL: llvm.func @callee(!llvm.ptr<i8>) -> i32
func.func private @callee(!test.test_type) -> i32

// CHECK-NEXT: llvm.func @caller() -> i32
func.func @caller() -> i32 {
  %arg = "test.type_producer"() : () -> !test.test_type
  %out = call @callee(%arg) : (!test.test_type) -> i32
  return %out : i32
}
// CHECK-NEXT: [[ARG:%.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK-NEXT: [[OUT:%.*]] = llvm.call @callee([[ARG]])
// CHECK-SAME:     : (!llvm.ptr<i8>) -> i32
