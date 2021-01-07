// RUN: mlir-opt -convert-spirv-to-llvm %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spv.ShiftRightArithmetic
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @shift_right_arithmetic_scalar
spv.func @shift_right_arithmetic_scalar(%arg0: i32, %arg1: si32, %arg2 : i16, %arg3 : ui16) "None" {
  // CHECK: llvm.ashr %{{.*}}, %{{.*}} : i32
  %0 = spv.ShiftRightArithmetic %arg0, %arg0 : i32, i32

  // CHECK: llvm.ashr %{{.*}}, %{{.*}} : i32
  %1 = spv.ShiftRightArithmetic %arg0, %arg1 : i32, si32

  // CHECK: %[[SEXT:.*]] = llvm.sext %{{.*}} : i16 to i32
  // CHECK: llvm.ashr %{{.*}}, %[[SEXT]] : i32
  %2 = spv.ShiftRightArithmetic %arg0, %arg2 : i32, i16

  // CHECK: %[[ZEXT:.*]] = llvm.zext %{{.*}} : i16 to i32
  // CHECK: llvm.ashr %{{.*}}, %[[ZEXT]] : i32
  %3 = spv.ShiftRightArithmetic %arg0, %arg3 : i32, ui16
  spv.Return
}

// CHECK-LABEL: @shift_right_arithmetic_vector
spv.func @shift_right_arithmetic_vector(%arg0: vector<4xi64>, %arg1: vector<4xui64>, %arg2: vector<4xi32>, %arg3: vector<4xui32>) "None" {
  // CHECK: llvm.ashr %{{.*}}, %{{.*}} : !llvm.vec<4 x i64>
  %0 = spv.ShiftRightArithmetic %arg0, %arg0 : vector<4xi64>, vector<4xi64>

  // CHECK: llvm.ashr %{{.*}}, %{{.*}} : !llvm.vec<4 x i64>
  %1 = spv.ShiftRightArithmetic %arg0, %arg1 : vector<4xi64>, vector<4xui64>

  // CHECK: %[[SEXT:.*]] = llvm.sext %{{.*}} : !llvm.vec<4 x i32> to !llvm.vec<4 x i64>
  // CHECK: llvm.ashr %{{.*}}, %[[SEXT]] : !llvm.vec<4 x i64>
  %2 = spv.ShiftRightArithmetic %arg0, %arg2 : vector<4xi64>,  vector<4xi32>

  // CHECK: %[[ZEXT:.*]] = llvm.zext %{{.*}} : !llvm.vec<4 x i32> to !llvm.vec<4 x i64>
  // CHECK: llvm.ashr %{{.*}}, %[[ZEXT]] : !llvm.vec<4 x i64>
  %3 = spv.ShiftRightArithmetic %arg0, %arg3 : vector<4xi64>, vector<4xui32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.ShiftRightLogical
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @shift_right_logical_scalar
spv.func @shift_right_logical_scalar(%arg0: i32, %arg1: si32, %arg2 : si16, %arg3 : ui16) "None" {
  // CHECK: llvm.lshr %{{.*}}, %{{.*}} : i32
  %0 = spv.ShiftRightLogical %arg0, %arg0 : i32, i32

  // CHECK: llvm.lshr %{{.*}}, %{{.*}} : i32
  %1 = spv.ShiftRightLogical %arg0, %arg1 : i32, si32

  // CHECK: %[[SEXT:.*]] = llvm.sext %{{.*}} : i16 to i32
  // CHECK: llvm.lshr %{{.*}}, %[[SEXT]] : i32
  %2 = spv.ShiftRightLogical %arg0, %arg2 : i32, si16

  // CHECK: %[[ZEXT:.*]] = llvm.zext %{{.*}} : i16 to i32
  // CHECK: llvm.lshr %{{.*}}, %[[ZEXT]] : i32
  %3 = spv.ShiftRightLogical %arg0, %arg3 : i32, ui16
  spv.Return
}

// CHECK-LABEL: @shift_right_logical_vector
spv.func @shift_right_logical_vector(%arg0: vector<4xi64>, %arg1: vector<4xsi64>, %arg2: vector<4xi32>, %arg3: vector<4xui32>) "None" {
  // CHECK: llvm.lshr %{{.*}}, %{{.*}} : !llvm.vec<4 x i64>
  %0 = spv.ShiftRightLogical %arg0, %arg0 : vector<4xi64>, vector<4xi64>

  // CHECK: llvm.lshr %{{.*}}, %{{.*}} : !llvm.vec<4 x i64>
  %1 = spv.ShiftRightLogical %arg0, %arg1 : vector<4xi64>, vector<4xsi64>

  // CHECK: %[[SEXT:.*]] = llvm.sext %{{.*}} : !llvm.vec<4 x i32> to !llvm.vec<4 x i64>
  // CHECK: llvm.lshr %{{.*}}, %[[SEXT]] : !llvm.vec<4 x i64>
  %2 = spv.ShiftRightLogical %arg0, %arg2 : vector<4xi64>,  vector<4xi32>

  // CHECK: %[[ZEXT:.*]] = llvm.zext %{{.*}} : !llvm.vec<4 x i32> to !llvm.vec<4 x i64>
  // CHECK: llvm.lshr %{{.*}}, %[[ZEXT]] : !llvm.vec<4 x i64>
  %3 = spv.ShiftRightLogical %arg0, %arg3 : vector<4xi64>, vector<4xui32>
  spv.Return
}

//===----------------------------------------------------------------------===//
// spv.ShiftLeftLogical
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @shift_left_logical_scalar
spv.func @shift_left_logical_scalar(%arg0: i32, %arg1: si32, %arg2 : i16, %arg3 : ui16) "None" {
  // CHECK: llvm.shl %{{.*}}, %{{.*}} : i32
  %0 = spv.ShiftLeftLogical %arg0, %arg0 : i32, i32

  // CHECK: llvm.shl %{{.*}}, %{{.*}} : i32
  %1 = spv.ShiftLeftLogical %arg0, %arg1 : i32, si32

  // CHECK: %[[SEXT:.*]] = llvm.sext %{{.*}} : i16 to i32
  // CHECK: llvm.shl %{{.*}}, %[[SEXT]] : i32
  %2 = spv.ShiftLeftLogical %arg0, %arg2 : i32, i16

  // CHECK: %[[ZEXT:.*]] = llvm.zext %{{.*}} : i16 to i32
  // CHECK: llvm.shl %{{.*}}, %[[ZEXT]] : i32
  %3 = spv.ShiftLeftLogical %arg0, %arg3 : i32, ui16
  spv.Return
}

// CHECK-LABEL: @shift_left_logical_vector
spv.func @shift_left_logical_vector(%arg0: vector<4xi64>, %arg1: vector<4xsi64>, %arg2: vector<4xi32>, %arg3: vector<4xui32>) "None" {
  // CHECK: llvm.shl %{{.*}}, %{{.*}} : !llvm.vec<4 x i64>
  %0 = spv.ShiftLeftLogical %arg0, %arg0 : vector<4xi64>, vector<4xi64>

  // CHECK: llvm.shl %{{.*}}, %{{.*}} : !llvm.vec<4 x i64>
  %1 = spv.ShiftLeftLogical %arg0, %arg1 : vector<4xi64>, vector<4xsi64>

  // CHECK: %[[SEXT:.*]] = llvm.sext %{{.*}} : !llvm.vec<4 x i32> to !llvm.vec<4 x i64>
  // CHECK: llvm.shl %{{.*}}, %[[SEXT]] : !llvm.vec<4 x i64>
  %2 = spv.ShiftLeftLogical %arg0, %arg2 : vector<4xi64>,  vector<4xi32>

  // CHECK: %[[ZEXT:.*]] = llvm.zext %{{.*}} : !llvm.vec<4 x i32> to !llvm.vec<4 x i64>
  // CHECK: llvm.shl %{{.*}}, %[[ZEXT]] : !llvm.vec<4 x i64>
  %3 = spv.ShiftLeftLogical %arg0, %arg3 : vector<4xi64>, vector<4xui32>
  spv.Return
}
