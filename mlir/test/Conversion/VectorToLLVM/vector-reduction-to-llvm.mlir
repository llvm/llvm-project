// RUN: mlir-opt %s -convert-vector-to-llvm | FileCheck %s
// RUN: mlir-opt %s -convert-vector-to-llvm='reassociate-fp-reductions' | FileCheck %s --check-prefix=REASSOC

//
// CHECK-LABEL: llvm.func @reduce_add_f32(
// CHECK-SAME: %[[A:.*]]: vector<16xf32>)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
// CHECK-SAME: {reassoc = false} : (f32, vector<16xf32>) -> f32
//      CHECK: llvm.return %[[V]] : f32
//
// REASSOC-LABEL: llvm.func @reduce_add_f32(
// REASSOC-SAME: %[[A:.*]]: vector<16xf32>)
//      REASSOC: %[[C:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
//      REASSOC: %[[V:.*]] = "llvm.intr.vector.reduce.fadd"(%[[C]], %[[A]])
// REASSOC-SAME: {reassoc = true} : (f32, vector<16xf32>) -> f32
//      REASSOC: llvm.return %[[V]] : f32
//
func @reduce_add_f32(%arg0: vector<16xf32>) -> f32 {
  %0 = vector.reduction "add", %arg0 : vector<16xf32> into f32
  return %0 : f32
}

//
// CHECK-LABEL: llvm.func @reduce_mul_f32(
// CHECK-SAME: %[[A:.*]]: vector<16xf32>)
//      CHECK: %[[C:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
//      CHECK: %[[V:.*]] = "llvm.intr.vector.reduce.fmul"(%[[C]], %[[A]])
// CHECK-SAME: {reassoc = false} : (f32, vector<16xf32>) -> f32
//      CHECK: llvm.return %[[V]] : f32
//
// REASSOC-LABEL: llvm.func @reduce_mul_f32(
// REASSOC-SAME: %[[A:.*]]: vector<16xf32>)
//      REASSOC: %[[C:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
//      REASSOC: %[[V:.*]] = "llvm.intr.vector.reduce.fmul"(%[[C]], %[[A]])
// REASSOC-SAME: {reassoc = true} : (f32, vector<16xf32>) -> f32
//      REASSOC: llvm.return %[[V]] : f32
//
func @reduce_mul_f32(%arg0: vector<16xf32>) -> f32 {
  %0 = vector.reduction "mul", %arg0 : vector<16xf32> into f32
  return %0 : f32
}
