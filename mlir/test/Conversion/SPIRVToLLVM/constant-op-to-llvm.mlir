// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s
// RUN: mlir-opt -convert-spirv-to-llvm='allow-pattern-rollback=0' %s | FileCheck %s

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

// CHECK-LABEL: @constant_with_discardable_attr
spirv.func @constant_with_discardable_attr() "None" {
  // CHECK: llvm.mlir.constant(42 : i32) {test.marker = "keep"} : i32
  %0 = "spirv.Constant"() <{value = 42 : i32}> {test.marker = "keep"} : () -> i32
  spirv.Return
}

// CHECK-LABEL: @array_constant
spirv.func @array_constant() "None" {
  // CHECK: llvm.mlir.constant(dense<[1, -2]> : tensor<2xi32>) : !llvm.array<2 x i32>
  %0 = spirv.Constant [1 : si32, -2 : si32] : !spirv.array<2 x si32>
  // CHECK: llvm.mlir.constant(dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>) : !llvm.array<2 x array<2 x f32>>
  %1 = spirv.Constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32> : !spirv.array<2 x !spirv.array<2 x f32>>
  // CHECK: %[[POISON:.*]] = llvm.mlir.poison : !llvm.array<2 x vector<2xi32>>
  // CHECK: %[[E0:.*]] = llvm.mlir.constant(dense<[1, 2]> : vector<2xi32>) : vector<2xi32>
  // CHECK: %[[A0:.*]] = llvm.insertvalue %[[E0]], %[[POISON]][0] : !llvm.array<2 x vector<2xi32>>
  // CHECK: %[[E1:.*]] = llvm.mlir.constant(dense<3> : vector<2xi32>) : vector<2xi32>
  // CHECK: llvm.insertvalue %[[E1]], %[[A0]][1] : !llvm.array<2 x vector<2xi32>>
  %2 = spirv.Constant [dense<[1, 2]> : vector<2xui32>, dense<3> : vector<2xui32>] : !spirv.array<2 x vector<2xui32>>
  // CHECK: llvm.mlir.constant(dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf8E4M3FN>) : !llvm.array<2 x i8>
  %3 = spirv.Constant [1.0 : f8E4M3FN, 2.0 : f8E4M3FN] : !spirv.array<2 x f8E4M3FN>
  spirv.Return
}

// CHECK-LABEL: @struct_constant
spirv.func @struct_constant() "None" {
  // CHECK: %[[POISON:.*]] = llvm.mlir.poison : !llvm.struct<packed (i8, array<2 x i32>, ptr)>
  // CHECK: %[[E0:.*]] = llvm.mlir.constant(-1 : i8) : i8
  // CHECK: %[[S0:.*]] = llvm.insertvalue %[[E0]], %[[POISON]][0]
  // CHECK: %[[E1:.*]] = llvm.mlir.constant(dense<[2, 3]> : tensor<2xi32>) : !llvm.array<2 x i32>
  // CHECK: %[[S1:.*]] = llvm.insertvalue %[[E1]], %[[S0]][1]
  // CHECK: %[[E2:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: llvm.insertvalue %[[E2]], %[[S1]][2]
  %0 = spirv.Constant [-1 : si8, [2 : i32, 3 : i32], #spirv.null : !spirv.ptr<f32, Function>] : !spirv.struct<(si8, !spirv.array<2 x i32>, !spirv.ptr<f32, Function>)>
  // CHECK: %[[F8:.*]] = llvm.mlir.constant(1.000000e+00 : f8E4M3FN) : i8
  // CHECK: llvm.insertvalue %[[F8]], %{{.*}}[0] : !llvm.struct<packed (i8)>
  %1 = spirv.Constant [1.0 : f8E4M3FN] : !spirv.struct<(f8E4M3FN)>
  // CHECK: llvm.mlir.zero : !llvm.struct<packed ()>
  %2 = spirv.Constant [] : !spirv.struct<()>
  spirv.Return
}
