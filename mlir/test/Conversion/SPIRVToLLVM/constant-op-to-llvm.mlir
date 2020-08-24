// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.constant
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bool_constant_scalar
spv.func @bool_constant_scalar() "None" {
  // CHECK: llvm.mlir.constant(true) : !llvm.i1
  %0 = spv.constant true
  // CHECK: llvm.mlir.constant(false) : !llvm.i1
  %1 = spv.constant false
  spv.Return
}

// CHECK-LABEL: @bool_constant_vector
spv.func @bool_constant_vector() "None" {
  // CHECK: llvm.mlir.constant(dense<[true, false]> : vector<2xi1>) : !llvm.vec<2 x i1>
  %0 = spv.constant dense<[true, false]> : vector<2xi1>
  // CHECK: llvm.mlir.constant(dense<false> : vector<3xi1>) : !llvm.vec<3 x i1>
  %1 = spv.constant dense<false> : vector<3xi1>
  spv.Return
}

// CHECK-LABEL: @integer_constant_scalar
spv.func @integer_constant_scalar() "None" {
  // CHECK: llvm.mlir.constant(0 : i8) : !llvm.i8
  %0 = spv.constant  0 : i8
  // CHECK: llvm.mlir.constant(-5 : i64) : !llvm.i64
  %1 = spv.constant -5 : si64
  // CHECK: llvm.mlir.constant(10 : i16) : !llvm.i16
  %2 = spv.constant  10 : ui16
  spv.Return
}

// CHECK-LABEL: @integer_constant_vector
spv.func @integer_constant_vector() "None" {
  // CHECK: llvm.mlir.constant(dense<[2, 3]> : vector<2xi32>) : !llvm.vec<2 x i32>
  %0 = spv.constant dense<[2, 3]> : vector<2xi32>
  // CHECK: llvm.mlir.constant(dense<-4> : vector<2xi32>) : !llvm.vec<2 x i32>
  %1 = spv.constant dense<-4> : vector<2xsi32>
  // CHECK: llvm.mlir.constant(dense<[2, 3, 4]> : vector<3xi32>) : !llvm.vec<3 x i32>
  %2 = spv.constant dense<[2, 3, 4]> : vector<3xui32>
  spv.Return
}

// CHECK-LABEL: @float_constant_scalar
spv.func @float_constant_scalar() "None" {
  // CHECK: llvm.mlir.constant(5.000000e+00 : f16) : !llvm.half
  %0 = spv.constant 5.000000e+00 : f16
  // CHECK: llvm.mlir.constant(5.000000e+00 : f64) : !llvm.double
  %1 = spv.constant 5.000000e+00 : f64
  spv.Return
}

// CHECK-LABEL: @float_constant_vector
spv.func @float_constant_vector() "None" {
  // CHECK: llvm.mlir.constant(dense<[2.000000e+00, 3.000000e+00]> : vector<2xf32>) : !llvm.vec<2 x float>
  %0 = spv.constant dense<[2.000000e+00, 3.000000e+00]> : vector<2xf32>
  spv.Return
}
