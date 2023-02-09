// RUN: mlir-opt -convert-func-to-llvm='use-opaque-pointers=1' %s | FileCheck %s

// CHECK-LABEL: func @check_attributes
// CHECK-SAME: {dialect.a = true, dialect.b = 4 : i64}
func.func @check_attributes(%int: i64 {dialect.a = true, dialect.b = 4 : i64 }) {
  return
}

// CHECK-LABEL: func @check_memref
// When expanding the memref to multiple arguments, argument attributes should be dropped entirely.
// CHECK-NOT: {llvm.noalias}
func.func @check_memref(%static: memref<10x20xf32> {llvm.noalias}) {
  return
}

// CHECK-LABEL: func @check_multiple
// CHECK-SAME: %{{.*}}: f32 {first.arg = true}, %{{.*}}: i64 {second.arg = 42 : i32}
func.func @check_multiple(%first: f32 {first.arg = true}, %second: i64 {second.arg = 42 : i32}) {
  return
}
