// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.Constant
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bool_constant_scalar
spv.func @bool_constant_scalar() "None" {
  // CHECK: llvm.mlir.constant(true) : i1
  %0 = spv.Constant true
  // CHECK: llvm.mlir.constant(false) : i1
  %1 = spv.Constant false
  spv.Return
}

// CHECK-LABEL: @bool_constant_vector
spv.func @bool_constant_vector() "None" {
  // CHECK: llvm.mlir.constant(dense<[true, false]> : vector<2xi1>) : vector<2xi1>
  %0 = spv.Constant dense<[true, false]> : vector<2xi1>
  // CHECK: llvm.mlir.constant(dense<false> : vector<3xi1>) : vector<3xi1>
  %1 = spv.Constant dense<false> : vector<3xi1>
  spv.Return
}

// CHECK-LABEL: @integer_constant_scalar
spv.func @integer_constant_scalar() "None" {
  // CHECK: llvm.mlir.constant(0 : i8) : i8
  %0 = spv.Constant  0 : i8
  // CHECK: llvm.mlir.constant(-5 : i64) : i64
  %1 = spv.Constant -5 : si64
  // CHECK: llvm.mlir.constant(10 : i16) : i16
  %2 = spv.Constant  10 : ui16
  spv.Return
}

// CHECK-LABEL: @integer_constant_vector
spv.func @integer_constant_vector() "None" {
  // CHECK: llvm.mlir.constant(dense<[2, 3]> : vector<2xi32>) : vector<2xi32>
  %0 = spv.Constant dense<[2, 3]> : vector<2xi32>
  // CHECK: llvm.mlir.constant(dense<-4> : vector<2xi32>) : vector<2xi32>
  %1 = spv.Constant dense<-4> : vector<2xsi32>
  // CHECK: llvm.mlir.constant(dense<[2, 3, 4]> : vector<3xi32>) : vector<3xi32>
  %2 = spv.Constant dense<[2, 3, 4]> : vector<3xui32>
  spv.Return
}

// CHECK-LABEL: @float_constant_scalar
spv.func @float_constant_scalar() "None" {
  // CHECK: llvm.mlir.constant(5.000000e+00 : f16) : f16
  %0 = spv.Constant 5.000000e+00 : f16
  // CHECK: llvm.mlir.constant(5.000000e+00 : f64) : f64
  %1 = spv.Constant 5.000000e+00 : f64
  spv.Return
}

// CHECK-LABEL: @float_constant_vector
spv.func @float_constant_vector() "None" {
  // CHECK: llvm.mlir.constant(dense<[2.000000e+00, 3.000000e+00]> : vector<2xf32>) : vector<2xf32>
  %0 = spv.Constant dense<[2.000000e+00, 3.000000e+00]> : vector<2xf32>
  spv.Return
}
