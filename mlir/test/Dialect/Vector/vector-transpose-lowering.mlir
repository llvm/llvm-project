// RUN: mlir-opt %s --test-transform-dialect-interpreter --split-input-file | FileCheck %s

// CHECK-LABEL: func @transpose23
// CHECK-SAME: %[[A:.*]]: vector<2x3xf32>
// CHECK:      %[[Z:.*]] = arith.constant dense<0.000000e+00> : vector<3x2xf32>
// CHECK:      %[[T0:.*]] = vector.extract %[[A]][0, 0] : f32 from vector<2x3xf32>
// CHECK:      %[[T1:.*]] = vector.insert %[[T0]], %[[Z]] [0, 0] : f32 into vector<3x2xf32>
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

transform.sequence failures(propagate) {
^bb1(%func_op: !transform.op<"func.func">):
  transform.apply_patterns to %func_op {
    transform.apply_patterns.vector.lower_transpose lowering_strategy = "eltwise"
  } : !transform.op<"func.func">
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


transform.sequence failures(propagate) {
^bb1(%func_op: !transform.op<"func.func">):
  transform.apply_patterns to %func_op {
    transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d"
  } : !transform.op<"func.func">
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


transform.sequence failures(propagate) {
^bb1(%func_op: !transform.op<"func.func">):
  transform.apply_patterns to %func_op {
    transform.apply_patterns.vector.lower_transpose lowering_strategy = "flat_transpose"
  } : !transform.op<"func.func">
}

// -----

// CHECK-LABEL: func @transpose4x8
func.func @transpose4x8xf32(%arg0: vector<4x8xf32>) -> vector<8x4xf32> {
  //      CHECK: vector.extract {{.*}}[0]
  // CHECK-NEXT: vector.extract {{.*}}[1]
  // CHECK-NEXT: vector.extract {{.*}}[2]
  // CHECK-NEXT: vector.extract {{.*}}[3]
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.insert {{.*}}[0]
  // CHECK-NEXT: vector.insert {{.*}}[1]
  // CHECK-NEXT: vector.insert {{.*}}[2]
  // CHECK-NEXT: vector.insert {{.*}}[3]
  // CHECK-NEXT: vector.shape_cast {{.*}} vector<4x8xf32> to vector<32xf32>
  // CHECK-NEXT: vector.shape_cast {{.*}} vector<32xf32> to vector<8x4xf32>
  %0 = vector.transpose %arg0, [1, 0] : vector<4x8xf32> to vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// CHECK-LABEL: func @transpose021_1x4x8
func.func @transpose021_1x4x8xf32(%arg0: vector<1x4x8xf32>) -> vector<1x8x4xf32> {
  //      CHECK: vector.extract {{.*}}[0, 0]
  // CHECK-NEXT: vector.extract {{.*}}[0, 1]
  // CHECK-NEXT: vector.extract {{.*}}[0, 2]
  // CHECK-NEXT: vector.extract {{.*}}[0, 3]
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.insert {{.*}}[0]
  // CHECK-NEXT: vector.insert {{.*}}[1]
  // CHECK-NEXT: vector.insert {{.*}}[2]
  // CHECK-NEXT: vector.insert {{.*}}[3]
  // CHECK-NEXT: vector.shape_cast {{.*}} vector<4x8xf32> to vector<32xf32>
  // CHECK-NEXT: vector.shape_cast {{.*}} vector<32xf32> to vector<1x8x4xf32>
  %0 = vector.transpose %arg0, [0, 2, 1] : vector<1x4x8xf32> to vector<1x8x4xf32>
  return %0 : vector<1x8x4xf32>
}

// CHECK-LABEL: func @transpose8x8
func.func @transpose8x8xf32(%arg0: vector<8x8xf32>) -> vector<8x8xf32> {
  //      CHECK: vector.extract {{.*}}[0]
  // CHECK-NEXT: vector.extract {{.*}}[1]
  // CHECK-NEXT: vector.extract {{.*}}[2]
  // CHECK-NEXT: vector.extract {{.*}}[3]
  // CHECK-NEXT: vector.extract {{.*}}[4]
  // CHECK-NEXT: vector.extract {{.*}}[5]
  // CHECK-NEXT: vector.extract {{.*}}[6]
  // CHECK-NEXT: vector.extract {{.*}}[7]
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.insert {{.*}}[0]
  // CHECK-NEXT: vector.insert {{.*}}[1]
  // CHECK-NEXT: vector.insert {{.*}}[2]
  // CHECK-NEXT: vector.insert {{.*}}[3]
  // CHECK-NEXT: vector.insert {{.*}}[4]
  // CHECK-NEXT: vector.insert {{.*}}[5]
  // CHECK-NEXT: vector.insert {{.*}}[6]
  // CHECK-NEXT: vector.insert {{.*}}[7]
  %0 = vector.transpose %arg0, [1, 0] : vector<8x8xf32> to vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// CHECK-LABEL: func @transpose021_1x8x8
func.func @transpose021_1x8x8xf32(%arg0: vector<1x8x8xf32>) -> vector<1x8x8xf32> {
  //      CHECK: vector.extract {{.*}}[0, 0]
  // CHECK-NEXT: vector.extract {{.*}}[0, 1]
  // CHECK-NEXT: vector.extract {{.*}}[0, 2]
  // CHECK-NEXT: vector.extract {{.*}}[0, 3]
  // CHECK-NEXT: vector.extract {{.*}}[0, 4]
  // CHECK-NEXT: vector.extract {{.*}}[0, 5]
  // CHECK-NEXT: vector.extract {{.*}}[0, 6]
  // CHECK-NEXT: vector.extract {{.*}}[0, 7]
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.insert {{.*}}[0]
  // CHECK-NEXT: vector.insert {{.*}}[1]
  // CHECK-NEXT: vector.insert {{.*}}[2]
  // CHECK-NEXT: vector.insert {{.*}}[3]
  // CHECK-NEXT: vector.insert {{.*}}[4]
  // CHECK-NEXT: vector.insert {{.*}}[5]
  // CHECK-NEXT: vector.insert {{.*}}[6]
  // CHECK-NEXT: vector.insert {{.*}}[7]
  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<1x8x8xf32>
  %0 = vector.transpose %arg0, [0, 2, 1] : vector<1x8x8xf32> to vector<1x8x8xf32>
  return %0 : vector<1x8x8xf32>
}

// CHECK-LABEL: func @transpose120_8x1x8
func.func @transpose120_8x1x8xf32(%arg0: vector<8x1x8xf32>) -> vector<1x8x8xf32> {
  //      CHECK: vector.extract {{.*}}[0, 0]
  // CHECK-NEXT: vector.extract {{.*}}[1, 0]
  // CHECK-NEXT: vector.extract {{.*}}[2, 0]
  // CHECK-NEXT: vector.extract {{.*}}[3, 0]
  // CHECK-NEXT: vector.extract {{.*}}[4, 0]
  // CHECK-NEXT: vector.extract {{.*}}[5, 0]
  // CHECK-NEXT: vector.extract {{.*}}[6, 0]
  // CHECK-NEXT: vector.extract {{.*}}[7, 0]
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.insert {{.*}}[0]
  // CHECK-NEXT: vector.insert {{.*}}[1]
  // CHECK-NEXT: vector.insert {{.*}}[2]
  // CHECK-NEXT: vector.insert {{.*}}[3]
  // CHECK-NEXT: vector.insert {{.*}}[4]
  // CHECK-NEXT: vector.insert {{.*}}[5]
  // CHECK-NEXT: vector.insert {{.*}}[6]
  // CHECK-NEXT: vector.insert {{.*}}[7]
  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<1x8x8xf32>
  %0 = vector.transpose %arg0, [1, 2, 0] : vector<8x1x8xf32> to vector<1x8x8xf32>
  return %0 : vector<1x8x8xf32>
}

// CHECK-LABEL: func @transpose120_8x8x1
func.func @transpose120_8x8x1xf32(%arg0: vector<8x8x1xf32>) -> vector<8x1x8xf32> {
  //      CHECK: vector.shape_cast %{{.*}} : vector<8x8x1xf32> to vector<8x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[0]
  // CHECK-NEXT: vector.extract {{.*}}[1]
  // CHECK-NEXT: vector.extract {{.*}}[2]
  // CHECK-NEXT: vector.extract {{.*}}[3]
  // CHECK-NEXT: vector.extract {{.*}}[4]
  // CHECK-NEXT: vector.extract {{.*}}[5]
  // CHECK-NEXT: vector.extract {{.*}}[6]
  // CHECK-NEXT: vector.extract {{.*}}[7]
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.insert {{.*}}[0]
  // CHECK-NEXT: vector.insert {{.*}}[1]
  // CHECK-NEXT: vector.insert {{.*}}[2]
  // CHECK-NEXT: vector.insert {{.*}}[3]
  // CHECK-NEXT: vector.insert {{.*}}[4]
  // CHECK-NEXT: vector.insert {{.*}}[5]
  // CHECK-NEXT: vector.insert {{.*}}[6]
  // CHECK-NEXT: vector.insert {{.*}}[7]
  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<8x1x8xf32>
  %0 = vector.transpose %arg0, [1, 2, 0] : vector<8x8x1xf32> to vector<8x1x8xf32>
  return %0 : vector<8x1x8xf32>
}

// CHECK-LABEL: func @transpose102_8x8x1
func.func @transpose102_8x8x1xf32(%arg0: vector<8x8x1xf32>) -> vector<8x8x1xf32> {
  //      CHECK: vector.shape_cast %{{.*}} : vector<8x8x1xf32> to vector<8x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[0]
  // CHECK-NEXT: vector.extract {{.*}}[1]
  // CHECK-NEXT: vector.extract {{.*}}[2]
  // CHECK-NEXT: vector.extract {{.*}}[3]
  // CHECK-NEXT: vector.extract {{.*}}[4]
  // CHECK-NEXT: vector.extract {{.*}}[5]
  // CHECK-NEXT: vector.extract {{.*}}[6]
  // CHECK-NEXT: vector.extract {{.*}}[7]
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.insert {{.*}}[0]
  // CHECK-NEXT: vector.insert {{.*}}[1]
  // CHECK-NEXT: vector.insert {{.*}}[2]
  // CHECK-NEXT: vector.insert {{.*}}[3]
  // CHECK-NEXT: vector.insert {{.*}}[4]
  // CHECK-NEXT: vector.insert {{.*}}[5]
  // CHECK-NEXT: vector.insert {{.*}}[6]
  // CHECK-NEXT: vector.insert {{.*}}[7]
  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<8x8x1xf32>
  %0 = vector.transpose %arg0, [1, 0, 2] : vector<8x8x1xf32> to vector<8x8x1xf32>
  return %0 : vector<8x8x1xf32>
}

// CHECK-LABEL: func @transpose201_8x1x8
func.func @transpose201_8x1x8xf32(%arg0: vector<8x1x8xf32>) -> vector<8x8x1xf32> {
  //      CHECK: vector.extract {{.*}}[0, 0]
  // CHECK-NEXT: vector.extract {{.*}}[1, 0]
  // CHECK-NEXT: vector.extract {{.*}}[2, 0]
  // CHECK-NEXT: vector.extract {{.*}}[3, 0]
  // CHECK-NEXT: vector.extract {{.*}}[4, 0]
  // CHECK-NEXT: vector.extract {{.*}}[5, 0]
  // CHECK-NEXT: vector.extract {{.*}}[6, 0]
  // CHECK-NEXT: vector.extract {{.*}}[7, 0]
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.insert {{.*}}[0]
  // CHECK-NEXT: vector.insert {{.*}}[1]
  // CHECK-NEXT: vector.insert {{.*}}[2]
  // CHECK-NEXT: vector.insert {{.*}}[3]
  // CHECK-NEXT: vector.insert {{.*}}[4]
  // CHECK-NEXT: vector.insert {{.*}}[5]
  // CHECK-NEXT: vector.insert {{.*}}[6]
  // CHECK-NEXT: vector.insert {{.*}}[7]
  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<8x8x1xf32>
  %0 = vector.transpose %arg0, [2, 0, 1] : vector<8x1x8xf32> to vector<8x8x1xf32>
  return %0 : vector<8x8x1xf32>
}

// CHECK-LABEL: func @transpose201_1x8x8
func.func @transpose201_1x8x8xf32(%arg0: vector<1x8x8xf32>) -> vector<8x1x8xf32> {
  //      CHECK: vector.extract {{.*}}[0, 0]
  // CHECK-NEXT: vector.extract {{.*}}[0, 1]
  // CHECK-NEXT: vector.extract {{.*}}[0, 2]
  // CHECK-NEXT: vector.extract {{.*}}[0, 3]
  // CHECK-NEXT: vector.extract {{.*}}[0, 4]
  // CHECK-NEXT: vector.extract {{.*}}[0, 5]
  // CHECK-NEXT: vector.extract {{.*}}[0, 6]
  // CHECK-NEXT: vector.extract {{.*}}[0, 7]
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.insert {{.*}}[0]
  // CHECK-NEXT: vector.insert {{.*}}[1]
  // CHECK-NEXT: vector.insert {{.*}}[2]
  // CHECK-NEXT: vector.insert {{.*}}[3]
  // CHECK-NEXT: vector.insert {{.*}}[4]
  // CHECK-NEXT: vector.insert {{.*}}[5]
  // CHECK-NEXT: vector.insert {{.*}}[6]
  // CHECK-NEXT: vector.insert {{.*}}[7]
  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<8x1x8xf32>
  %0 = vector.transpose %arg0, [2, 0, 1] : vector<1x8x8xf32> to vector<8x1x8xf32>
  return %0 : vector<8x1x8xf32>
}

// CHECK-LABEL: func @transpose210_8x1x8
func.func @transpose210_8x1x8xf32(%arg0: vector<8x1x8xf32>) -> vector<8x1x8xf32> {
  //      CHECK: vector.extract {{.*}}[0, 0]
  // CHECK-NEXT: vector.extract {{.*}}[1, 0]
  // CHECK-NEXT: vector.extract {{.*}}[2, 0]
  // CHECK-NEXT: vector.extract {{.*}}[3, 0]
  // CHECK-NEXT: vector.extract {{.*}}[4, 0]
  // CHECK-NEXT: vector.extract {{.*}}[5, 0]
  // CHECK-NEXT: vector.extract {{.*}}[6, 0]
  // CHECK-NEXT: vector.extract {{.*}}[7, 0]
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.insert {{.*}}[0]
  // CHECK-NEXT: vector.insert {{.*}}[1]
  // CHECK-NEXT: vector.insert {{.*}}[2]
  // CHECK-NEXT: vector.insert {{.*}}[3]
  // CHECK-NEXT: vector.insert {{.*}}[4]
  // CHECK-NEXT: vector.insert {{.*}}[5]
  // CHECK-NEXT: vector.insert {{.*}}[6]
  // CHECK-NEXT: vector.insert {{.*}}[7]
  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<8x1x8xf32>
  %0 = vector.transpose %arg0, [2, 1, 0] : vector<8x1x8xf32> to vector<8x1x8xf32>
  return %0 : vector<8x1x8xf32>
}

// CHECK-LABEL: func @transpose210_8x8x1
func.func @transpose210_8x8x1xf32(%arg0: vector<8x8x1xf32>) -> vector<1x8x8xf32> {
  //      CHECK: vector.shape_cast %{{.*}} : vector<8x8x1xf32> to vector<8x8xf32>
  // CHECK-NEXT: vector.extract {{.*}}[0]
  // CHECK-NEXT: vector.extract {{.*}}[1]
  // CHECK-NEXT: vector.extract {{.*}}[2]
  // CHECK-NEXT: vector.extract {{.*}}[3]
  // CHECK-NEXT: vector.extract {{.*}}[4]
  // CHECK-NEXT: vector.extract {{.*}}[5]
  // CHECK-NEXT: vector.extract {{.*}}[6]
  // CHECK-NEXT: vector.extract {{.*}}[7]
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.insert {{.*}}[0]
  // CHECK-NEXT: vector.insert {{.*}}[1]
  // CHECK-NEXT: vector.insert {{.*}}[2]
  // CHECK-NEXT: vector.insert {{.*}}[3]
  // CHECK-NEXT: vector.insert {{.*}}[4]
  // CHECK-NEXT: vector.insert {{.*}}[5]
  // CHECK-NEXT: vector.insert {{.*}}[6]
  // CHECK-NEXT: vector.insert {{.*}}[7]
  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<1x8x8xf32>
  %0 = vector.transpose %arg0, [2, 1, 0] : vector<8x8x1xf32> to vector<1x8x8xf32>
  return %0 : vector<1x8x8xf32>
}

// CHECK-LABEL: func @transpose210_1x8x8
func.func @transpose210_1x8x8xf32(%arg0: vector<1x8x8xf32>) -> vector<8x8x1xf32> {
  //      CHECK: vector.extract {{.*}}[0, 0]
  // CHECK-NEXT: vector.extract {{.*}}[0, 1]
  // CHECK-NEXT: vector.extract {{.*}}[0, 2]
  // CHECK-NEXT: vector.extract {{.*}}[0, 3]
  // CHECK-NEXT: vector.extract {{.*}}[0, 4]
  // CHECK-NEXT: vector.extract {{.*}}[0, 5]
  // CHECK-NEXT: vector.extract {{.*}}[0, 6]
  // CHECK-NEXT: vector.extract {{.*}}[0, 7]
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // CHECK-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // CHECK-NEXT: vector.insert {{.*}}[0]
  // CHECK-NEXT: vector.insert {{.*}}[1]
  // CHECK-NEXT: vector.insert {{.*}}[2]
  // CHECK-NEXT: vector.insert {{.*}}[3]
  // CHECK-NEXT: vector.insert {{.*}}[4]
  // CHECK-NEXT: vector.insert {{.*}}[5]
  // CHECK-NEXT: vector.insert {{.*}}[6]
  // CHECK-NEXT: vector.insert {{.*}}[7]
  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<8x8x1xf32>
  %0 = vector.transpose %arg0, [2, 1, 0] : vector<1x8x8xf32> to vector<8x8x1xf32>
  return %0 : vector<8x8x1xf32>
}

transform.sequence failures(propagate) {
^bb1(%func_op: !transform.op<"func.func">):
  transform.apply_patterns to %func_op {
    transform.apply_patterns.vector.lower_transpose avx2_lowering_strategy = true
  } : !transform.op<"func.func">
}

// -----

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

transform.sequence failures(propagate) {
^bb1(%func_op: !transform.op<"func.func">):
  transform.apply_patterns to %func_op {
    transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_16x16"
  } : !transform.op<"func.func">
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

transform.sequence failures(propagate) {
^bb1(%func_op: !transform.op<"func.func">):
  transform.apply_patterns to %func_op {
    transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_16x16"
  } : !transform.op<"func.func">
}
