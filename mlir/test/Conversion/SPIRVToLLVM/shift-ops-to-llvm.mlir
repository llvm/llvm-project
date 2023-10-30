// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.ShiftRightArithmetic
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @shift_right_arithmetic_scalar
spirv.func @shift_right_arithmetic_scalar(%arg0: i32, %arg1: si32, %arg2 : i16, %arg3 : ui16) "None" {
  // CHECK: llvm.ashr %{{.*}}, %{{.*}} : i32
  %0 = spirv.ShiftRightArithmetic %arg0, %arg0 : i32, i32

  // CHECK: llvm.ashr %{{.*}}, %{{.*}} : i32
  %1 = spirv.ShiftRightArithmetic %arg0, %arg1 : i32, si32

  // CHECK: %[[SEXT:.*]] = llvm.sext %{{.*}} : i16 to i32
  // CHECK: llvm.ashr %{{.*}}, %[[SEXT]] : i32
  %2 = spirv.ShiftRightArithmetic %arg0, %arg2 : i32, i16

  // CHECK: %[[ZEXT:.*]] = llvm.zext %{{.*}} : i16 to i32
  // CHECK: llvm.ashr %{{.*}}, %[[ZEXT]] : i32
  %3 = spirv.ShiftRightArithmetic %arg0, %arg3 : i32, ui16
  spirv.Return
}

// CHECK-LABEL: @shift_right_arithmetic_vector
spirv.func @shift_right_arithmetic_vector(%arg0: vector<4xi64>, %arg1: vector<4xui64>, %arg2: vector<4xi32>, %arg3: vector<4xui32>) "None" {
  // CHECK: llvm.ashr %{{.*}}, %{{.*}} : vector<4xi64>
  %0 = spirv.ShiftRightArithmetic %arg0, %arg0 : vector<4xi64>, vector<4xi64>

  // CHECK: llvm.ashr %{{.*}}, %{{.*}} : vector<4xi64>
  %1 = spirv.ShiftRightArithmetic %arg0, %arg1 : vector<4xi64>, vector<4xui64>

  // CHECK: %[[SEXT:.*]] = llvm.sext %{{.*}} : vector<4xi32> to vector<4xi64>
  // CHECK: llvm.ashr %{{.*}}, %[[SEXT]] : vector<4xi64>
  %2 = spirv.ShiftRightArithmetic %arg0, %arg2 : vector<4xi64>,  vector<4xi32>

  // CHECK: %[[ZEXT:.*]] = llvm.zext %{{.*}} : vector<4xi32> to vector<4xi64>
  // CHECK: llvm.ashr %{{.*}}, %[[ZEXT]] : vector<4xi64>
  %3 = spirv.ShiftRightArithmetic %arg0, %arg3 : vector<4xi64>, vector<4xui32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.ShiftRightLogical
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @shift_right_logical_scalar
spirv.func @shift_right_logical_scalar(%arg0: i32, %arg1: si32, %arg2 : si16, %arg3 : ui16) "None" {
  // CHECK: llvm.lshr %{{.*}}, %{{.*}} : i32
  %0 = spirv.ShiftRightLogical %arg0, %arg0 : i32, i32

  // CHECK: llvm.lshr %{{.*}}, %{{.*}} : i32
  %1 = spirv.ShiftRightLogical %arg0, %arg1 : i32, si32

  // CHECK: %[[SEXT:.*]] = llvm.sext %{{.*}} : i16 to i32
  // CHECK: llvm.lshr %{{.*}}, %[[SEXT]] : i32
  %2 = spirv.ShiftRightLogical %arg0, %arg2 : i32, si16

  // CHECK: %[[ZEXT:.*]] = llvm.zext %{{.*}} : i16 to i32
  // CHECK: llvm.lshr %{{.*}}, %[[ZEXT]] : i32
  %3 = spirv.ShiftRightLogical %arg0, %arg3 : i32, ui16
  spirv.Return
}

// CHECK-LABEL: @shift_right_logical_vector
spirv.func @shift_right_logical_vector(%arg0: vector<4xi64>, %arg1: vector<4xsi64>, %arg2: vector<4xi32>, %arg3: vector<4xui32>) "None" {
  // CHECK: llvm.lshr %{{.*}}, %{{.*}} : vector<4xi64>
  %0 = spirv.ShiftRightLogical %arg0, %arg0 : vector<4xi64>, vector<4xi64>

  // CHECK: llvm.lshr %{{.*}}, %{{.*}} : vector<4xi64>
  %1 = spirv.ShiftRightLogical %arg0, %arg1 : vector<4xi64>, vector<4xsi64>

  // CHECK: %[[SEXT:.*]] = llvm.sext %{{.*}} : vector<4xi32> to vector<4xi64>
  // CHECK: llvm.lshr %{{.*}}, %[[SEXT]] : vector<4xi64>
  %2 = spirv.ShiftRightLogical %arg0, %arg2 : vector<4xi64>,  vector<4xi32>

  // CHECK: %[[ZEXT:.*]] = llvm.zext %{{.*}} : vector<4xi32> to vector<4xi64>
  // CHECK: llvm.lshr %{{.*}}, %[[ZEXT]] : vector<4xi64>
  %3 = spirv.ShiftRightLogical %arg0, %arg3 : vector<4xi64>, vector<4xui32>
  spirv.Return
}

//===----------------------------------------------------------------------===//
// spirv.ShiftLeftLogical
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @shift_left_logical_scalar
spirv.func @shift_left_logical_scalar(%arg0: i32, %arg1: si32, %arg2 : i16, %arg3 : ui16) "None" {
  // CHECK: llvm.shl %{{.*}}, %{{.*}} : i32
  %0 = spirv.ShiftLeftLogical %arg0, %arg0 : i32, i32

  // CHECK: llvm.shl %{{.*}}, %{{.*}} : i32
  %1 = spirv.ShiftLeftLogical %arg0, %arg1 : i32, si32

  // CHECK: %[[SEXT:.*]] = llvm.sext %{{.*}} : i16 to i32
  // CHECK: llvm.shl %{{.*}}, %[[SEXT]] : i32
  %2 = spirv.ShiftLeftLogical %arg0, %arg2 : i32, i16

  // CHECK: %[[ZEXT:.*]] = llvm.zext %{{.*}} : i16 to i32
  // CHECK: llvm.shl %{{.*}}, %[[ZEXT]] : i32
  %3 = spirv.ShiftLeftLogical %arg0, %arg3 : i32, ui16
  spirv.Return
}

// CHECK-LABEL: @shift_left_logical_vector
spirv.func @shift_left_logical_vector(%arg0: vector<4xi64>, %arg1: vector<4xsi64>, %arg2: vector<4xi32>, %arg3: vector<4xui32>) "None" {
  // CHECK: llvm.shl %{{.*}}, %{{.*}} : vector<4xi64>
  %0 = spirv.ShiftLeftLogical %arg0, %arg0 : vector<4xi64>, vector<4xi64>

  // CHECK: llvm.shl %{{.*}}, %{{.*}} : vector<4xi64>
  %1 = spirv.ShiftLeftLogical %arg0, %arg1 : vector<4xi64>, vector<4xsi64>

  // CHECK: %[[SEXT:.*]] = llvm.sext %{{.*}} : vector<4xi32> to vector<4xi64>
  // CHECK: llvm.shl %{{.*}}, %[[SEXT]] : vector<4xi64>
  %2 = spirv.ShiftLeftLogical %arg0, %arg2 : vector<4xi64>,  vector<4xi32>

  // CHECK: %[[ZEXT:.*]] = llvm.zext %{{.*}} : vector<4xi32> to vector<4xi64>
  // CHECK: llvm.shl %{{.*}}, %[[ZEXT]] : vector<4xi64>
  %3 = spirv.ShiftLeftLogical %arg0, %arg3 : vector<4xi64>, vector<4xui32>
  spirv.Return
}
