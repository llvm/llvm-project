// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.Constant
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bool_constant_scalar
spirv.func @bool_constant_scalar() "None" {
  // CHECK: llvm.mlir.constant(true) : i1
  %0 = spirv.Constant true
  // CHECK: llvm.mlir.constant(false) : i1
  %1 = spirv.Constant false
  spirv.Return
}

// CHECK-LABEL: @bool_constant_vector
spirv.func @bool_constant_vector() "None" {
  // CHECK: llvm.mlir.constant(dense<[true, false]> : vector<2xi1>) : vector<2xi1>
  %0 = spirv.Constant dense<[true, false]> : vector<2xi1>
  // CHECK: llvm.mlir.constant(dense<false> : vector<3xi1>) : vector<3xi1>
  %1 = spirv.Constant dense<false> : vector<3xi1>
  spirv.Return
}

// CHECK-LABEL: @integer_constant_scalar
spirv.func @integer_constant_scalar() "None" {
  // CHECK: llvm.mlir.constant(0 : i8) : i8
  %0 = spirv.Constant  0 : i8
  // CHECK: llvm.mlir.constant(-5 : i64) : i64
  %1 = spirv.Constant -5 : si64
  // CHECK: llvm.mlir.constant(10 : i16) : i16
  %2 = spirv.Constant  10 : ui16
  spirv.Return
}

// CHECK-LABEL: @integer_constant_vector
spirv.func @integer_constant_vector() "None" {
  // CHECK: llvm.mlir.constant(dense<[2, 3]> : vector<2xi32>) : vector<2xi32>
  %0 = spirv.Constant dense<[2, 3]> : vector<2xi32>
  // CHECK: llvm.mlir.constant(dense<-4> : vector<2xi32>) : vector<2xi32>
  %1 = spirv.Constant dense<-4> : vector<2xsi32>
  // CHECK: llvm.mlir.constant(dense<[2, 3, 4]> : vector<3xi32>) : vector<3xi32>
  %2 = spirv.Constant dense<[2, 3, 4]> : vector<3xui32>
  spirv.Return
}

// CHECK-LABEL: @float_constant_scalar
spirv.func @float_constant_scalar() "None" {
  // CHECK: llvm.mlir.constant(5.000000e+00 : f16) : f16
  %0 = spirv.Constant 5.000000e+00 : f16
  // CHECK: llvm.mlir.constant(5.000000e+00 : f64) : f64
  %1 = spirv.Constant 5.000000e+00 : f64
  spirv.Return
}

// CHECK-LABEL: @float_constant_vector
spirv.func @float_constant_vector() "None" {
  // CHECK: llvm.mlir.constant(dense<[2.000000e+00, 3.000000e+00]> : vector<2xf32>) : vector<2xf32>
  %0 = spirv.Constant dense<[2.000000e+00, 3.000000e+00]> : vector<2xf32>
  spirv.Return
}
