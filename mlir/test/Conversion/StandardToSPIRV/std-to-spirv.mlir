// RUN: mlir-opt -convert-std-to-spirv %s -o - | FileCheck %s

//===----------------------------------------------------------------------===//
// std binary arithmetic ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @add_sub
func @add_sub(%arg0 : i32, %arg1 : i32) {
  // CHECK: spv.IAdd
  %0 = addi %arg0, %arg1 : i32
  // CHECK: spv.ISub
  %1 = subi %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @fadd_scalar
func @fadd_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FAdd
  %0 = addf %arg, %arg : f32
  return %0 : f32
}

// CHECK-LABEL: @fdiv_scalar
func @fdiv_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FDiv
  %0 = divf %arg, %arg : f32
  return %0 : f32
}

// CHECK-LABEL: @fmul_scalar
func @fmul_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FMul
  %0 = mulf %arg, %arg : f32
  return %0 : f32
}

// CHECK-LABEL: @fmul_vector2
func @fmul_vector2(%arg: vector<2xf32>) -> vector<2xf32> {
  // CHECK: spv.FMul
  %0 = mulf %arg, %arg : vector<2xf32>
  return %0 : vector<2xf32>
}

// CHECK-LABEL: @fmul_vector3
func @fmul_vector3(%arg: vector<3xf32>) -> vector<3xf32> {
  // CHECK: spv.FMul
  %0 = mulf %arg, %arg : vector<3xf32>
  return %0 : vector<3xf32>
}

// CHECK-LABEL: @fmul_vector4
func @fmul_vector4(%arg: vector<4xf32>) -> vector<4xf32> {
  // CHECK: spv.FMul
  %0 = mulf %arg, %arg : vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: @fmul_vector5
func @fmul_vector5(%arg: vector<5xf32>) -> vector<5xf32> {
  // Vector length of only 2, 3, and 4 is valid for SPIR-V
  // CHECK: mulf
  %0 = mulf %arg, %arg : vector<5xf32>
  return %0 : vector<5xf32>
}

// CHECK-LABEL: @fmul_tensor
func @fmul_tensor(%arg: tensor<4xf32>) -> tensor<4xf32> {
  // For tensors mulf cannot be lowered directly to spv.FMul
  // CHECK: mulf
  %0 = mulf %arg, %arg : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: @frem_scalar
func @frem_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FRem
  %0 = remf %arg, %arg : f32
  return %0 : f32
}

// CHECK-LABEL: @fsub_scalar
func @fsub_scalar(%arg: f32) -> f32 {
  // CHECK: spv.FSub
  %0 = subf %arg, %arg : f32
  return %0 : f32
}

// CHECK-LABEL: @div_rem
func @div_rem(%arg0 : i32, %arg1 : i32) {
  // CHECK: spv.SDiv
  %0 = divi_signed %arg0, %arg1 : i32
  // CHECK: spv.SMod
  %1 = remi_signed %arg0, %arg1 : i32
  return
}

//===----------------------------------------------------------------------===//
// std.cmpi
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cmpi
func @cmpi(%arg0 : i32, %arg1 : i32) {
  // CHECK: spv.IEqual
  %0 = cmpi "eq", %arg0, %arg1 : i32
  // CHECK: spv.INotEqual
  %1 = cmpi "ne", %arg0, %arg1 : i32
  // CHECK: spv.SLessThan
  %2 = cmpi "slt", %arg0, %arg1 : i32
  // CHECK: spv.SLessThanEqual
  %3 = cmpi "sle", %arg0, %arg1 : i32
  // CHECK: spv.SGreaterThan
  %4 = cmpi "sgt", %arg0, %arg1 : i32
  // CHECK: spv.SGreaterThanEqual
  %5 = cmpi "sge", %arg0, %arg1 : i32
  return
}

//===----------------------------------------------------------------------===//
// std.constant
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @constant
func @constant() {
  // CHECK: spv.constant true
  %0 = constant true
  // CHECK: spv.constant 42 : i64
  %1 = constant 42
  // CHECK: spv.constant {{[0-9]*\.[0-9]*e?-?[0-9]*}} : f32
  %2 = constant 0.5 : f32
  // CHECK: spv.constant dense<[2, 3]> : vector<2xi32>
  %3 = constant dense<[2, 3]> : vector<2xi32>
  // CHECK: spv.constant 1 : i32
  %4 = constant 1 : index
  return
}

//===----------------------------------------------------------------------===//
// std logical binary operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @logical_scalar
func @logical_scalar(%arg0 : i1, %arg1 : i1) {
  // CHECK: spv.LogicalAnd
  %0 = and %arg0, %arg1 : i1
  // CHECK: spv.LogicalOr
  %1 = or %arg0, %arg1 : i1
  return
}

// CHECK-LABEL: @logical_vector
func @logical_vector(%arg0 : vector<4xi1>, %arg1 : vector<4xi1>) {
  // CHECK: spv.LogicalAnd
  %0 = and %arg0, %arg1 : vector<4xi1>
  // CHECK: spv.LogicalOr
  %1 = or %arg0, %arg1 : vector<4xi1>
  return
}

// CHECK-LABEL: @logical_scalar_fail
func @logical_scalar_fail(%arg0 : i32, %arg1 : i32) {
  // CHECK-NOT: spv.LogicalAnd
  %0 = and %arg0, %arg1 : i32
  // CHECK-NOT: spv.LogicalOr
  %1 = or %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @logical_vector_fail
func @logical_vector_fail(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
  // CHECK-NOT: spv.LogicalAnd
  %0 = and %arg0, %arg1 : vector<4xi32>
  // CHECK-NOT: spv.LogicalOr
  %1 = or %arg0, %arg1 : vector<4xi32>
  return
}

//===----------------------------------------------------------------------===//
// std.select
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @select
func @select(%arg0 : i32, %arg1 : i32) {
  %0 = cmpi "sle", %arg0, %arg1 : i32
  // CHECK: spv.Select
  %1 = select %0, %arg0, %arg1 : i32
  return
}
