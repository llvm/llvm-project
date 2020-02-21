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
func @fadd_scalar(%arg: f32) {
  // CHECK: spv.FAdd
  %0 = addf %arg, %arg : f32
  return
}

// CHECK-LABEL: @fdiv_scalar
func @fdiv_scalar(%arg: f32) {
  // CHECK: spv.FDiv
  %0 = divf %arg, %arg : f32
  return
}

// CHECK-LABEL: @fmul_scalar
func @fmul_scalar(%arg: f32) {
  // CHECK: spv.FMul
  %0 = mulf %arg, %arg : f32
  return
}

// CHECK-LABEL: @fmul_vector2
func @fmul_vector2(%arg: vector<2xf32>) {
  // CHECK: spv.FMul
  %0 = mulf %arg, %arg : vector<2xf32>
  return
}

// CHECK-LABEL: @fmul_vector3
func @fmul_vector3(%arg: vector<3xf32>) {
  // CHECK: spv.FMul
  %0 = mulf %arg, %arg : vector<3xf32>
  return
}

// CHECK-LABEL: @fmul_vector4
func @fmul_vector4(%arg: vector<4xf32>) {
  // CHECK: spv.FMul
  %0 = mulf %arg, %arg : vector<4xf32>
  return
}

// CHECK-LABEL: @fmul_vector5
func @fmul_vector5(%arg: vector<5xf32>) {
  // Vector length of only 2, 3, and 4 is valid for SPIR-V.
  // CHECK: mulf
  %0 = mulf %arg, %arg : vector<5xf32>
  return
}

// TODO(antiagainst): enable this once we support converting binary ops
// needing type conversion.
// XXXXX-LABEL: @fmul_tensor
//func @fmul_tensor(%arg: tensor<4xf32>) {
  // For tensors mulf cannot be lowered directly to spv.FMul.
  // XXXXX: mulf
  //%0 = mulf %arg, %arg : tensor<4xf32>
  //return
//}

// CHECK-LABEL: @frem_scalar
func @frem_scalar(%arg: f32) {
  // CHECK: spv.FRem
  %0 = remf %arg, %arg : f32
  return
}

// CHECK-LABEL: @fsub_scalar
func @fsub_scalar(%arg: f32) {
  // CHECK: spv.FSub
  %0 = subf %arg, %arg : f32
  return
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
// std bit ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @bitwise_scalar
func @bitwise_scalar(%arg0 : i32, %arg1 : i32) {
  // CHECK: spv.BitwiseAnd
  %0 = and %arg0, %arg1 : i32
  // CHECK: spv.BitwiseOr
  %1 = or %arg0, %arg1 : i32
  // CHECK: spv.BitwiseXor
  %2 = xor %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @bitwise_vector
func @bitwise_vector(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
  // CHECK: spv.BitwiseAnd
  %0 = and %arg0, %arg1 : vector<4xi32>
  // CHECK: spv.BitwiseOr
  %1 = or %arg0, %arg1 : vector<4xi32>
  // CHECK: spv.BitwiseXor
  %2 = xor %arg0, %arg1 : vector<4xi32>
  return
}

// CHECK-LABEL: @shift_scalar
func @shift_scalar(%arg0 : i32, %arg1 : i32) {
  // CHECK: spv.ShiftLeftLogical
  %0 = shift_left %arg0, %arg1 : i32
  // CHECK: spv.ShiftRightArithmetic
  %1 = shift_right_signed %arg0, %arg1 : i32
  // CHECK: spv.ShiftRightLogical
  %2 = shift_right_unsigned %arg0, %arg1 : i32
  return
}

// CHECK-LABEL: @shift_vector
func @shift_vector(%arg0 : vector<4xi32>, %arg1 : vector<4xi32>) {
  // CHECK: spv.ShiftLeftLogical
  %0 = shift_left %arg0, %arg1 : vector<4xi32>
  // CHECK: spv.ShiftRightArithmetic
  %1 = shift_right_signed %arg0, %arg1 : vector<4xi32>
  // CHECK: spv.ShiftRightLogical
  %2 = shift_right_unsigned %arg0, %arg1 : vector<4xi32>
  return
}

//===----------------------------------------------------------------------===//
// std.cmpf
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @cmpf
func @cmpf(%arg0 : f32, %arg1 : f32) {
  // CHECK: spv.FOrdEqual
  %1 = cmpf "oeq", %arg0, %arg1 : f32
  // CHECK: spv.FOrdGreaterThan
  %2 = cmpf "ogt", %arg0, %arg1 : f32
  // CHECK: spv.FOrdGreaterThanEqual
  %3 = cmpf "oge", %arg0, %arg1 : f32
  // CHECK: spv.FOrdLessThan
  %4 = cmpf "olt", %arg0, %arg1 : f32
  // CHECK: spv.FOrdLessThanEqual
  %5 = cmpf "ole", %arg0, %arg1 : f32
  // CHECK: spv.FOrdNotEqual
  %6 = cmpf "one", %arg0, %arg1 : f32
  // CHECK: spv.FUnordEqual
  %7 = cmpf "ueq", %arg0, %arg1 : f32
  // CHECK: spv.FUnordGreaterThan
  %8 = cmpf "ugt", %arg0, %arg1 : f32
  // CHECK: spv.FUnordGreaterThanEqual
  %9 = cmpf "uge", %arg0, %arg1 : f32
  // CHECK: spv.FUnordLessThan
  %10 = cmpf "ult", %arg0, %arg1 : f32
  // CHECK: FUnordLessThanEqual
  %11 = cmpf "ule", %arg0, %arg1 : f32
  // CHECK: spv.FUnordNotEqual
  %12 = cmpf "une", %arg0, %arg1 : f32
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
  // CHECK: spv.ULessThan
  %6 = cmpi "ult", %arg0, %arg1 : i32
  // CHECK: spv.ULessThanEqual
  %7 = cmpi "ule", %arg0, %arg1 : i32
  // CHECK: spv.UGreaterThan
  %8 = cmpi "ugt", %arg0, %arg1 : i32
  // CHECK: spv.UGreaterThanEqual
  %9 = cmpi "uge", %arg0, %arg1 : i32
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
  // CHECK: spv.constant dense<1> : tensor<6xi32> : !spv.array<6 x i32 [4]>
  %5 = constant dense<1> : tensor<2x3xi32>
  // CHECK: spv.constant dense<1.000000e+00> : tensor<6xf32> : !spv.array<6 x f32 [4]>
  %6 = constant dense<1.0> : tensor<2x3xf32>
  // CHECK: spv.constant dense<{{\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf32> : !spv.array<6 x f32 [4]>
  %7 = constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  // CHECK: spv.constant dense<{{\[}}1, 2, 3, 4, 5, 6]> : tensor<6xi32> : !spv.array<6 x i32 [4]>
  %8 = constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  // CHECK: spv.constant dense<{{\[}}1, 2, 3, 4, 5, 6]> : tensor<6xi32> : !spv.array<6 x i32 [4]>
  %9 =  constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>
  // CHECK: spv.constant dense<{{\[}}1, 2, 3, 4, 5, 6]> : tensor<6xi32> : !spv.array<6 x i32 [4]>
  %10 =  constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>
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

//===----------------------------------------------------------------------===//
// std.fpext
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fpext
func @fpext(%arg0 : f32) {
  // CHECK: spv.FConvert
  %0 = std.fpext %arg0 : f32 to f64
  return
}

//===----------------------------------------------------------------------===//
// std.fptrunc
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @fptrunc
func @fptrunc(%arg0 : f64) {
  // CHECK: spv.FConvert
  %0 = std.fptrunc %arg0 : f64 to f32
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

//===----------------------------------------------------------------------===//
// std.sitofp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @sitofp
func @sitofp(%arg0 : i32) {
  // CHECK: spv.ConvertSToF
  %0 = std.sitofp %arg0 : i32 to f32
  return
}

//===----------------------------------------------------------------------===//
// memref type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @memref_type({{%.*}}: memref<3xi1>)
func @memref_type(%arg0: memref<3xi1>) {
  return
}

// CHECK-LABEL: @load_store_zero_rank_float
// CHECK: [[ARG0:%.*]]: !spv.ptr<!spv.struct<!spv.array<1 x f32 [4]> [0]>, StorageBuffer>,
// CHECK: [[ARG1:%.*]]: !spv.ptr<!spv.struct<!spv.array<1 x f32 [4]> [0]>, StorageBuffer>)
func @load_store_zero_rank_float(%arg0: memref<f32>, %arg1: memref<f32>) {
  //      CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
  //      CHECK: spv.AccessChain [[ARG0]][
  // CHECK-SAME: [[ZERO1]], [[ZERO1]]
  // CHECK-SAME: ] :
  //      CHECK: spv.Load "StorageBuffer" %{{.*}} : f32
  %0 = load %arg0[] : memref<f32>
  //      CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
  //      CHECK: spv.AccessChain [[ARG1]][
  // CHECK-SAME: [[ZERO2]], [[ZERO2]]
  // CHECK-SAME: ] :
  //      CHECK: spv.Store "StorageBuffer" %{{.*}} : f32
  store %0, %arg1[] : memref<f32>
  return
}

// CHECK-LABEL: @load_store_zero_rank_int
// CHECK: [[ARG0:%.*]]: !spv.ptr<!spv.struct<!spv.array<1 x i32 [4]> [0]>, StorageBuffer>,
// CHECK: [[ARG1:%.*]]: !spv.ptr<!spv.struct<!spv.array<1 x i32 [4]> [0]>, StorageBuffer>)
func @load_store_zero_rank_int(%arg0: memref<i32>, %arg1: memref<i32>) {
  //      CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
  //      CHECK: spv.AccessChain [[ARG0]][
  // CHECK-SAME: [[ZERO1]], [[ZERO1]]
  // CHECK-SAME: ] :
  //      CHECK: spv.Load "StorageBuffer" %{{.*}} : i32
  %0 = load %arg0[] : memref<i32>
  //      CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
  //      CHECK: spv.AccessChain [[ARG1]][
  // CHECK-SAME: [[ZERO2]], [[ZERO2]]
  // CHECK-SAME: ] :
  //      CHECK: spv.Store "StorageBuffer" %{{.*}} : i32
  store %0, %arg1[] : memref<i32>
  return
}
