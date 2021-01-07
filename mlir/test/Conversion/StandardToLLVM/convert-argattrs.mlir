// RUN: mlir-opt -convert-std-to-llvm %s | FileCheck %s

// CHECK-LABEL: func @check_attributes
// When expanding the memref to multiple arguments, argument attributes are replicated.
// CHECK-COUNT-7: {dialect.a = true, dialect.b = 4 : i64}
func @check_attributes(%static: memref<10x20xf32> {dialect.a = true, dialect.b = 4 : i64 }) {
  %c0 = constant 0 : index
  %0 = load %static[%c0, %c0]: memref<10x20xf32>
  return
}

// CHECK-LABEL: func @check_multiple
// Make sure arguments attributes are attached to the right argument. We match
// commas in the argument list for this purpose.
// CHECK: %{{.*}}: !llvm{{.*}} {first.arg = true}, %{{.*}}: !llvm{{.*}} {first.arg = true}, %{{.*}}: i{{.*}} {first.arg = true},
// CHECK-SAME: %{{.*}}: !llvm{{.*}} {second.arg = 42 : i32}, %{{.*}}: !llvm{{.*}} {second.arg = 42 : i32}, %{{.*}}: i{{.*}} {second.arg = 42 : i32})
func @check_multiple(%first: memref<f32> {first.arg = true}, %second: memref<f32> {second.arg = 42 : i32}) {
  return
}
