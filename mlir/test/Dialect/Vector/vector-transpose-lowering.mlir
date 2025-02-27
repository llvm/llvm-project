// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: func @transpose23
// CHECK-SAME: %[[A:.*]]: vector<2x3xf32>
// CHECK:      %[[UB:.*]] = ub.poison : vector<3x2xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0, 0] : f32 from vector<2x3xf32>
// CHECK:      %[[T1:.*]] = vector.insert %[[T0]], %[[UB]] [0, 0] : f32 into vector<3x2xf32>
// CHECK:      %[[T2:.*]] = vector.extract %[[A]][0, 1] : f32 from vector<2x3xf32>
// CHECK:      %[[T3:.*]] = vector.insert %[[T2]], %[[T1]] [1, 0] : f32 into vector<3x2xf32>
// CHECK:      %[[T4:.*]] = vector.extract %[[A]][0, 2] : f32 from vector<2x3xf32>
// CHECK:      %[[T5:.*]] = vector.insert %[[T4]], %[[T3]] [2, 0] : f32 into vector<3x2xf32>
// CHECK:      %[[T6:.*]] = vector.extract %[[A]][1, 0] : f32 from vector<2x3xf32>
// CHECK:      %[[T7:.*]] = vector.insert %[[T6]], %[[T5]] [0, 1] : f32 into vector<3x2xf32>
// CHECK:      %[[T8:.*]] = vector.extract %[[A]][1, 1] : f32 from vector<2x3xf32>
// CHECK:      %[[T9:.*]] = vector.insert %[[T8]], %[[T7]] [1, 1] : f32 into vector<3x2xf32>
// CHECK:      %[[T10:.*]] = vector.extract %[[A]][1, 2] : f32 from vector<2x3xf32>
// CHECK:      %[[T11:.*]] = vector.insert %[[T10]], %[[T9]] [2, 1] : f32 into vector<3x2xf32>
// CHECK:      return %[[T11]] : vector<3x2xf32>
func.func @transpose23(%arg0: vector<2x3xf32>) -> vector<3x2xf32> {
  %0 = vector.transpose %arg0, [1, 0] : vector<2x3xf32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

// CHECK-LABEL: func @transpose102_1x8x8xf32
func.func @transpose102_1x8x8xf32(%arg0: vector<1x8x8xf32>) -> vector<8x1x8xf32> {
  //      CHECK: vector.extract {{.*}}[0, 0] : vector<8xf32> from vector<1x8x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 0] : vector<8xf32> into vector<8x1x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[0, 1] : vector<8xf32> from vector<1x8x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [1, 0] : vector<8xf32> into vector<8x1x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[0, 2] : vector<8xf32> from vector<1x8x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [2, 0] : vector<8xf32> into vector<8x1x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[0, 3] : vector<8xf32> from vector<1x8x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [3, 0] : vector<8xf32> into vector<8x1x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[0, 4] : vector<8xf32> from vector<1x8x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [4, 0] : vector<8xf32> into vector<8x1x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[0, 5] : vector<8xf32> from vector<1x8x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [5, 0] : vector<8xf32> into vector<8x1x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[0, 6] : vector<8xf32> from vector<1x8x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [6, 0] : vector<8xf32> into vector<8x1x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[0, 7] : vector<8xf32> from vector<1x8x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [7, 0] : vector<8xf32> into vector<8x1x8xf32>
  %0 = vector.transpose %arg0, [1, 0, 2] : vector<1x8x8xf32> to vector<8x1x8xf32>
  return %0 : vector<8x1x8xf32>
}

// CHECK-LABEL: func @transpose102_8x1x8xf32
func.func @transpose102_8x1x8xf32(%arg0: vector<8x1x8xf32>) -> vector<1x8x8xf32> {
  //      CHECK: vector.extract {{.*}}[0, 0] : vector<8xf32> from vector<8x1x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 0] : vector<8xf32> into vector<1x8x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[1, 0] : vector<8xf32> from vector<8x1x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 1] : vector<8xf32> into vector<1x8x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[2, 0] : vector<8xf32> from vector<8x1x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 2] : vector<8xf32> into vector<1x8x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[3, 0] : vector<8xf32> from vector<8x1x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 3] : vector<8xf32> into vector<1x8x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[4, 0] : vector<8xf32> from vector<8x1x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 4] : vector<8xf32> into vector<1x8x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[5, 0] : vector<8xf32> from vector<8x1x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 5] : vector<8xf32> into vector<1x8x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[6, 0] : vector<8xf32> from vector<8x1x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 6] : vector<8xf32> into vector<1x8x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[7, 0] : vector<8xf32> from vector<8x1x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 7] : vector<8xf32> into vector<1x8x8xf32>
  %0 = vector.transpose %arg0, [1, 0, 2] : vector<8x1x8xf32> to vector<1x8x8xf32>
  return %0 : vector<1x8x8xf32>
}

// CHECK-LABEL:   func @transpose1023_1x1x8x8xf32(
func.func @transpose1023_1x1x8x8xf32(%arg0: vector<1x1x8x8xf32>) -> vector<1x1x8x8xf32> {
  // Note the single 2-D extract/insert pair since 2 and 3 are not transposed!
  //      CHECK: vector.extract {{.*}}[0, 0] : vector<8x8xf32> from vector<1x1x8x8xf32>
  // CHECK-NEXT: vector.insert {{.*}} [0, 0] : vector<8x8xf32> into vector<1x1x8x8xf32>
  %0 = vector.transpose %arg0, [1, 0, 2, 3] : vector<1x1x8x8xf32> to vector<1x1x8x8xf32>
  return %0 : vector<1x1x8x8xf32>
}

/// Scalable dim should not be unrolled.

// CHECK-LABEL: func @transpose23_scalable
// CHECK-NOT: vector.extract
// CHECK-NOT: vector.insert
// CHECK: vector.transpose
func.func @transpose23_scalable(%arg0: vector<2x[3]xf32>) -> vector<[3]x2xf32> {
  %0 = vector.transpose %arg0, [1, 0] : vector<2x[3]xf32> to vector<[3]x2xf32>
  return %0 : vector<[3]x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_transpose lowering_strategy = "eltwise"
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @transpose
func.func @transpose(%arg0: vector<2x4xf32>) -> vector<4x2xf32> {
  //      CHECK: vector.shape_cast %{{.*}} : vector<2x4xf32> to vector<8xf32>
  //            0 4
  // 0 1 2 3    1 5
  // 4 5 6 7 -> 2 6
  //            3 7
  // CHECK-NEXT: vector.shuffle %{{.*}} [0, 4, 1, 5, 2, 6, 3, 7] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<8xf32> to vector<4x2xf32>
  %0 = vector.transpose %arg0, [1, 0] : vector<2x4xf32> to vector<4x2xf32>
  return %0 : vector<4x2xf32>
}

/// Scalable vectors are not supported

// CHECK-LABEL: func @transpose_scalable
// CHECK-NOT: vector.shuffle
// CHECK-NOT: vector.shape_cast
// CHECK: vector.transpose
func.func @transpose_scalable(%arg0: vector<2x[4]xf32>) -> vector<[4]x2xf32> {
  %0 = vector.transpose %arg0, [1, 0] : vector<2x[4]xf32> to vector<[4]x2xf32>
  return %0 : vector<[4]x2xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d"
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @transpose(
func.func @transpose(%arg0: vector<2x4xf32>) -> vector<4x2xf32> {
  // CHECK:       vector.shape_cast {{.*}} : vector<2x4xf32> to vector<8xf32>
  // CHECK:       vector.flat_transpose %{{.*}} {columns = 2 : i32, rows = 4 : i32} : vector<8xf32> -> vector<8xf32>
  // CHECK:       vector.shape_cast {{.*}} : vector<8xf32> to vector<4x2xf32>
  %0 = vector.transpose %arg0, [1, 0] : vector<2x4xf32> to vector<4x2xf32>
  return %0 : vector<4x2xf32>
}

/// Scalable vectors are not supported

// CHECK-LABEL: func @transpose_scalable(
func.func @transpose_scalable(%arg0: vector<2x[4]xf32>) -> vector<[4]x2xf32> {
  // CHECK-NOT:       vector.shape_cast
  // CHECK-NOT:       vector.flat_transpose
  // CHECK:           vector.transpose
  %0 = vector.transpose %arg0, [1, 0] : vector<2x[4]xf32> to vector<[4]x2xf32>
  return %0 : vector<[4]x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_transpose lowering_strategy = "flat_transpose"
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

// CHECK-LABEL: @transpose_shuffle16x16xf32(
func.func @transpose_shuffle16x16xf32(%arg0: vector<16x16xf32>) -> vector<16x16xf32> {
  // CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  %0 = vector.transpose %arg0, [1, 0] : vector<16x16xf32> to vector<16x16xf32>
  return %0 : vector<16x16xf32>
}

// CHECK-LABEL: @transpose_shuffle16x16xf32_scalable(
func.func @transpose_shuffle16x16xf32_scalable(%arg0: vector<16x[16]xf32>) -> vector<[16]x16xf32> {
  // CHECK-NOT: vector.shuffle
  // CHECK: vector.transpose
  %0 = vector.transpose %arg0, [1, 0] : vector<16x[16]xf32> to vector<[16]x16xf32>
  return %0 : vector<[16]x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_16x16"
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @transpose021_shuffle16x16xf32
func.func @transpose021_shuffle16x16xf32(%arg0: vector<1x16x16xf32>) -> vector<1x16x16xf32> {
  // CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  // CHECK: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
  %0 = vector.transpose %arg0, [0, 2, 1] : vector<1x16x16xf32> to vector<1x16x16xf32>
  return %0 : vector<1x16x16xf32>
}

// CHECK-LABEL: func @transpose021_shuffle16x16xf32_scalable
func.func @transpose021_shuffle16x16xf32_scalable(%arg0: vector<1x16x[16]xf32>) -> vector<1x[16]x16xf32> {
  // CHECK-NOT: vector.shuffle
  // CHECK: vector.transpose
  %0 = vector.transpose %arg0, [0, 2, 1] : vector<1x16x[16]xf32> to vector<1x[16]x16xf32>
  return %0 : vector<1x[16]x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_16x16"
    } : !transform.op<"func.func">
    transform.yield
  }
}

// -----

/// Transpose of rank-2 vector with leading or trailing unit dim to shape_cast.

// CHECK-LABEL: func @transpose10_4x1xf32
func.func @transpose10_4x1xf32(%arg0: vector<4x1xf32>) -> vector<1x4xf32> {
  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<4x1xf32> to vector<1x4xf32>
  %0 = vector.transpose %arg0, [1, 0] : vector<4x1xf32> to vector<1x4xf32>
  return %0 : vector<1x4xf32>
}

// CHECK-LABEL: func @transpose10_nx4x1xf32
func.func @transpose10_nx4x1xf32(%arg0: vector<[4]x1xf32>) -> vector<1x[4]xf32> {
  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<[4]x1xf32> to vector<1x[4]xf32>
  %0 = vector.transpose %arg0, [1, 0] : vector<[4]x1xf32> to vector<1x[4]xf32>
  return %0 : vector<1x[4]xf32>
}

// CHECK-LABEL: func @transpose10_1x4xf32
func.func @transpose10_1x4xf32(%arg0: vector<1x4xf32>) -> vector<4x1xf32> {
  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<1x4xf32> to vector<4x1xf32>
  %0 = vector.transpose %arg0, [1, 0] : vector<1x4xf32> to vector<4x1xf32>
  return %0 : vector<4x1xf32>
}

// CHECK-LABEL: func @transpose10_1xnx4xf32
func.func @transpose10_1xnx4xf32(%arg0: vector<1x[4]xf32>) -> vector<[4]x1xf32> {
  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<1x[4]xf32> to vector<[4]x1xf32>
  %0 = vector.transpose %arg0, [1, 0] : vector<1x[4]xf32> to vector<[4]x1xf32>
  return %0 : vector<[4]x1xf32>
}

/// Scalable unit dim should not be lowered to shape_cast.

// CHECK-LABEL: func @transpose10_4x1xf32_scalable
func.func @transpose10_4x1xf32_scalable(%arg0: vector<4x[1]xf32>) -> vector<[1]x4xf32> {
  // CHECK-NOT: vector.shape_cast
  // CHECK: vector.transpose %{{.*}} : vector<4x[1]xf32> to vector<[1]x4xf32>
  %0 = vector.transpose %arg0, [1, 0] : vector<4x[1]xf32> to vector<[1]x4xf32>
  return %0 : vector<[1]x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_transpose
    } : !transform.op<"func.func">
    transform.yield
  }
}
