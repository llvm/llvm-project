// RUN: mlir-opt %s -test-vector-transpose-lowering=eltwise=1 -split-input-file | FileCheck %s --check-prefix=ELTWISE
// RUN: mlir-opt %s -test-vector-transpose-lowering=shuffle=1 -split-input-file | FileCheck %s --check-prefix=SHUFFLE
// RUN: mlir-opt %s -test-vector-transpose-lowering=flat=1 -split-input-file | FileCheck %s --check-prefix=FLAT
// RUN: mlir-opt %s -test-vector-transpose-lowering=avx2=1 -split-input-file | FileCheck %s --check-prefix=AVX2

// ELTWISE-LABEL: func @transpose23
// ELTWISE-SAME: %[[A:.*]]: vector<2x3xf32>
// ELTWISE:      %[[Z:.*]] = arith.constant dense<0.000000e+00> : vector<3x2xf32>
// ELTWISE:      %[[T0:.*]] = vector.extract %[[A]][0, 0] : vector<2x3xf32>
// ELTWISE:      %[[T1:.*]] = vector.insert %[[T0]], %[[Z]] [0, 0] : f32 into vector<3x2xf32>
// ELTWISE:      %[[T2:.*]] = vector.extract %[[A]][0, 1] : vector<2x3xf32>
// ELTWISE:      %[[T3:.*]] = vector.insert %[[T2]], %[[T1]] [1, 0] : f32 into vector<3x2xf32>
// ELTWISE:      %[[T4:.*]] = vector.extract %[[A]][0, 2] : vector<2x3xf32>
// ELTWISE:      %[[T5:.*]] = vector.insert %[[T4]], %[[T3]] [2, 0] : f32 into vector<3x2xf32>
// ELTWISE:      %[[T6:.*]] = vector.extract %[[A]][1, 0] : vector<2x3xf32>
// ELTWISE:      %[[T7:.*]] = vector.insert %[[T6]], %[[T5]] [0, 1] : f32 into vector<3x2xf32>
// ELTWISE:      %[[T8:.*]] = vector.extract %[[A]][1, 1] : vector<2x3xf32>
// ELTWISE:      %[[T9:.*]] = vector.insert %[[T8]], %[[T7]] [1, 1] : f32 into vector<3x2xf32>
// ELTWISE:      %[[T10:.*]] = vector.extract %[[A]][1, 2] : vector<2x3xf32>
// ELTWISE:      %[[T11:.*]] = vector.insert %[[T10]], %[[T9]] [2, 1] : f32 into vector<3x2xf32>
// ELTWISE:      return %[[T11]] : vector<3x2xf32>
func @transpose23(%arg0: vector<2x3xf32>) -> vector<3x2xf32> {
  %0 = vector.transpose %arg0, [1, 0] : vector<2x3xf32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

// -----

// SHUFFLE-LABEL: func @transpose
// FLAT-LABEL: func @transpose(
func @transpose(%arg0: vector<2x4xf32>) -> vector<4x2xf32> {
  //      SHUFFLE: vector.shape_cast %{{.*}} : vector<2x4xf32> to vector<8xf32>
  //            0 4
  // 0 1 2 3    1 5
  // 4 5 6 7 -> 2 6
  //            3 7
  // SHUFFLE-NEXT: vector.shuffle %{{.*}} [0, 4, 1, 5, 2, 6, 3, 7] : vector<8xf32>, vector<8xf32>
  // SHUFFLE-NEXT: vector.shape_cast %{{.*}} : vector<8xf32> to vector<4x2xf32>

  // FLAT:       vector.shape_cast {{.*}} : vector<2x4xf32> to vector<8xf32>
  // FLAT:       vector.flat_transpose %{{.*}} {columns = 2 : i32, rows = 4 : i32} : vector<8xf32> -> vector<8xf32>
  // FLAT:       vector.shape_cast {{.*}} : vector<8xf32> to vector<4x2xf32>
  %0 = vector.transpose %arg0, [1, 0] : vector<2x4xf32> to vector<4x2xf32>
  return %0 : vector<4x2xf32>
}

// -----

// AVX2-LABEL: func @transpose4x8
func @transpose4x8xf32(%arg0: vector<4x8xf32>) -> vector<8x4xf32> {
  //      AVX2: vector.extract {{.*}}[0]
  // AVX2-NEXT: vector.extract {{.*}}[1]
  // AVX2-NEXT: vector.extract {{.*}}[2]
  // AVX2-NEXT: vector.extract {{.*}}[3]
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.insert {{.*}}[0]
  // AVX2-NEXT: vector.insert {{.*}}[1]
  // AVX2-NEXT: vector.insert {{.*}}[2]
  // AVX2-NEXT: vector.insert {{.*}}[3]
  // AVX2-NEXT: vector.shape_cast {{.*}} vector<4x8xf32> to vector<32xf32>
  // AVX2-NEXT: vector.shape_cast {{.*}} vector<32xf32> to vector<8x4xf32>
  %0 = vector.transpose %arg0, [1, 0] : vector<4x8xf32> to vector<8x4xf32>
  return %0 : vector<8x4xf32>
}

// -----

// AVX2-LABEL: func @transpose021_1x4x8
func @transpose021_1x4x8xf32(%arg0: vector<1x4x8xf32>) -> vector<1x8x4xf32> {
  //      AVX2: vector.extract {{.*}}[0, 0]
  // AVX2-NEXT: vector.extract {{.*}}[0, 1]
  // AVX2-NEXT: vector.extract {{.*}}[0, 2]
  // AVX2-NEXT: vector.extract {{.*}}[0, 3]
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 1, 8, 9, 4, 5, 12, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 3, 10, 11, 6, 7, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.insert {{.*}}[0]
  // AVX2-NEXT: vector.insert {{.*}}[1]
  // AVX2-NEXT: vector.insert {{.*}}[2]
  // AVX2-NEXT: vector.insert {{.*}}[3]
  // AVX2-NEXT: vector.shape_cast {{.*}} vector<4x8xf32> to vector<32xf32>
  // AVX2-NEXT: vector.shape_cast {{.*}} vector<32xf32> to vector<1x8x4xf32>
  %0 = vector.transpose %arg0, [0, 2, 1] : vector<1x4x8xf32> to vector<1x8x4xf32>
  return %0 : vector<1x8x4xf32>
}

// -----

// AVX2-LABEL: func @transpose8x8
func @transpose8x8xf32(%arg0: vector<8x8xf32>) -> vector<8x8xf32> {
  //      AVX2: vector.extract {{.*}}[0]
  // AVX2-NEXT: vector.extract {{.*}}[1]
  // AVX2-NEXT: vector.extract {{.*}}[2]
  // AVX2-NEXT: vector.extract {{.*}}[3]
  // AVX2-NEXT: vector.extract {{.*}}[4]
  // AVX2-NEXT: vector.extract {{.*}}[5]
  // AVX2-NEXT: vector.extract {{.*}}[6]
  // AVX2-NEXT: vector.extract {{.*}}[7]
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.insert {{.*}}[0]
  // AVX2-NEXT: vector.insert {{.*}}[1]
  // AVX2-NEXT: vector.insert {{.*}}[2]
  // AVX2-NEXT: vector.insert {{.*}}[3]
  // AVX2-NEXT: vector.insert {{.*}}[4]
  // AVX2-NEXT: vector.insert {{.*}}[5]
  // AVX2-NEXT: vector.insert {{.*}}[6]
  // AVX2-NEXT: vector.insert {{.*}}[7]
  %0 = vector.transpose %arg0, [1, 0] : vector<8x8xf32> to vector<8x8xf32>
  return %0 : vector<8x8xf32>
}

// -----

// AVX2-LABEL: func @transpose021_1x8x8
func @transpose021_1x8x8xf32(%arg0: vector<1x8x8xf32>) -> vector<1x8x8xf32> {
  //      AVX2: vector.extract {{.*}}[0, 0]
  // AVX2-NEXT: vector.extract {{.*}}[0, 1]
  // AVX2-NEXT: vector.extract {{.*}}[0, 2]
  // AVX2-NEXT: vector.extract {{.*}}[0, 3]
  // AVX2-NEXT: vector.extract {{.*}}[0, 4]
  // AVX2-NEXT: vector.extract {{.*}}[0, 5]
  // AVX2-NEXT: vector.extract {{.*}}[0, 6]
  // AVX2-NEXT: vector.extract {{.*}}[0, 7]
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.insert {{.*}}[0]
  // AVX2-NEXT: vector.insert {{.*}}[1]
  // AVX2-NEXT: vector.insert {{.*}}[2]
  // AVX2-NEXT: vector.insert {{.*}}[3]
  // AVX2-NEXT: vector.insert {{.*}}[4]
  // AVX2-NEXT: vector.insert {{.*}}[5]
  // AVX2-NEXT: vector.insert {{.*}}[6]
  // AVX2-NEXT: vector.insert {{.*}}[7]
  // AVX2-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<1x8x8xf32>
  %0 = vector.transpose %arg0, [0, 2, 1] : vector<1x8x8xf32> to vector<1x8x8xf32>
  return %0 : vector<1x8x8xf32>
}

// -----

// AVX2-LABEL: func @transpose120_8x1x8
func @transpose120_8x1x8xf32(%arg0: vector<8x1x8xf32>) -> vector<1x8x8xf32> {
  //      AVX2: vector.extract {{.*}}[0, 0]
  // AVX2-NEXT: vector.extract {{.*}}[1, 0]
  // AVX2-NEXT: vector.extract {{.*}}[2, 0]
  // AVX2-NEXT: vector.extract {{.*}}[3, 0]
  // AVX2-NEXT: vector.extract {{.*}}[4, 0]
  // AVX2-NEXT: vector.extract {{.*}}[5, 0]
  // AVX2-NEXT: vector.extract {{.*}}[6, 0]
  // AVX2-NEXT: vector.extract {{.*}}[7, 0]
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.insert {{.*}}[0]
  // AVX2-NEXT: vector.insert {{.*}}[1]
  // AVX2-NEXT: vector.insert {{.*}}[2]
  // AVX2-NEXT: vector.insert {{.*}}[3]
  // AVX2-NEXT: vector.insert {{.*}}[4]
  // AVX2-NEXT: vector.insert {{.*}}[5]
  // AVX2-NEXT: vector.insert {{.*}}[6]
  // AVX2-NEXT: vector.insert {{.*}}[7]
  // AVX2-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<1x8x8xf32>
  %0 = vector.transpose %arg0, [1, 2, 0] : vector<8x1x8xf32> to vector<1x8x8xf32>
  return %0 : vector<1x8x8xf32>
}

// -----

// AVX2-LABEL: func @transpose120_8x8x1
func @transpose120_8x8x1xf32(%arg0: vector<8x8x1xf32>) -> vector<8x1x8xf32> {
  //      AVX2: vector.shape_cast %{{.*}} : vector<8x8x1xf32> to vector<8x8xf32>
  // AVX2-NEXT: vector.extract {{.*}}[0]
  // AVX2-NEXT: vector.extract {{.*}}[1]
  // AVX2-NEXT: vector.extract {{.*}}[2]
  // AVX2-NEXT: vector.extract {{.*}}[3]
  // AVX2-NEXT: vector.extract {{.*}}[4]
  // AVX2-NEXT: vector.extract {{.*}}[5]
  // AVX2-NEXT: vector.extract {{.*}}[6]
  // AVX2-NEXT: vector.extract {{.*}}[7]
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.insert {{.*}}[0]
  // AVX2-NEXT: vector.insert {{.*}}[1]
  // AVX2-NEXT: vector.insert {{.*}}[2]
  // AVX2-NEXT: vector.insert {{.*}}[3]
  // AVX2-NEXT: vector.insert {{.*}}[4]
  // AVX2-NEXT: vector.insert {{.*}}[5]
  // AVX2-NEXT: vector.insert {{.*}}[6]
  // AVX2-NEXT: vector.insert {{.*}}[7]
  // AVX2-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<8x1x8xf32>
  %0 = vector.transpose %arg0, [1, 2, 0] : vector<8x8x1xf32> to vector<8x1x8xf32>
  return %0 : vector<8x1x8xf32>
}

// -----

// AVX2-LABEL: func @transpose102_8x8x1
func @transpose102_8x8x1xf32(%arg0: vector<8x8x1xf32>) -> vector<8x8x1xf32> {
  //      AVX2: vector.shape_cast %{{.*}} : vector<8x8x1xf32> to vector<8x8xf32>
  // AVX2-NEXT: vector.extract {{.*}}[0]
  // AVX2-NEXT: vector.extract {{.*}}[1]
  // AVX2-NEXT: vector.extract {{.*}}[2]
  // AVX2-NEXT: vector.extract {{.*}}[3]
  // AVX2-NEXT: vector.extract {{.*}}[4]
  // AVX2-NEXT: vector.extract {{.*}}[5]
  // AVX2-NEXT: vector.extract {{.*}}[6]
  // AVX2-NEXT: vector.extract {{.*}}[7]
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.insert {{.*}}[0]
  // AVX2-NEXT: vector.insert {{.*}}[1]
  // AVX2-NEXT: vector.insert {{.*}}[2]
  // AVX2-NEXT: vector.insert {{.*}}[3]
  // AVX2-NEXT: vector.insert {{.*}}[4]
  // AVX2-NEXT: vector.insert {{.*}}[5]
  // AVX2-NEXT: vector.insert {{.*}}[6]
  // AVX2-NEXT: vector.insert {{.*}}[7]
  // AVX2-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<8x8x1xf32>
  %0 = vector.transpose %arg0, [1, 0, 2] : vector<8x8x1xf32> to vector<8x8x1xf32>
  return %0 : vector<8x8x1xf32>
}

// -----

// AVX2-LABEL: func @transpose201_8x1x8
func @transpose201_8x1x8xf32(%arg0: vector<8x1x8xf32>) -> vector<8x8x1xf32> {
  //      AVX2: vector.extract {{.*}}[0, 0]
  // AVX2-NEXT: vector.extract {{.*}}[1, 0]
  // AVX2-NEXT: vector.extract {{.*}}[2, 0]
  // AVX2-NEXT: vector.extract {{.*}}[3, 0]
  // AVX2-NEXT: vector.extract {{.*}}[4, 0]
  // AVX2-NEXT: vector.extract {{.*}}[5, 0]
  // AVX2-NEXT: vector.extract {{.*}}[6, 0]
  // AVX2-NEXT: vector.extract {{.*}}[7, 0]
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.insert {{.*}}[0]
  // AVX2-NEXT: vector.insert {{.*}}[1]
  // AVX2-NEXT: vector.insert {{.*}}[2]
  // AVX2-NEXT: vector.insert {{.*}}[3]
  // AVX2-NEXT: vector.insert {{.*}}[4]
  // AVX2-NEXT: vector.insert {{.*}}[5]
  // AVX2-NEXT: vector.insert {{.*}}[6]
  // AVX2-NEXT: vector.insert {{.*}}[7]
  // AVX2-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<8x8x1xf32>
  %0 = vector.transpose %arg0, [2, 0, 1] : vector<8x1x8xf32> to vector<8x8x1xf32>
  return %0 : vector<8x8x1xf32>
}

// -----

// AVX2-LABEL: func @transpose201_1x8x8
func @transpose201_1x8x8xf32(%arg0: vector<1x8x8xf32>) -> vector<8x1x8xf32> {
  //      AVX2: vector.extract {{.*}}[0, 0]
  // AVX2-NEXT: vector.extract {{.*}}[0, 1]
  // AVX2-NEXT: vector.extract {{.*}}[0, 2]
  // AVX2-NEXT: vector.extract {{.*}}[0, 3]
  // AVX2-NEXT: vector.extract {{.*}}[0, 4]
  // AVX2-NEXT: vector.extract {{.*}}[0, 5]
  // AVX2-NEXT: vector.extract {{.*}}[0, 6]
  // AVX2-NEXT: vector.extract {{.*}}[0, 7]
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.insert {{.*}}[0]
  // AVX2-NEXT: vector.insert {{.*}}[1]
  // AVX2-NEXT: vector.insert {{.*}}[2]
  // AVX2-NEXT: vector.insert {{.*}}[3]
  // AVX2-NEXT: vector.insert {{.*}}[4]
  // AVX2-NEXT: vector.insert {{.*}}[5]
  // AVX2-NEXT: vector.insert {{.*}}[6]
  // AVX2-NEXT: vector.insert {{.*}}[7]
  // AVX2-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<8x1x8xf32>
  %0 = vector.transpose %arg0, [2, 0, 1] : vector<1x8x8xf32> to vector<8x1x8xf32>
  return %0 : vector<8x1x8xf32>
}

// -----

// AVX2-LABEL: func @transpose210_8x1x8
func @transpose210_8x1x8xf32(%arg0: vector<8x1x8xf32>) -> vector<8x1x8xf32> {
  //      AVX2: vector.extract {{.*}}[0, 0]
  // AVX2-NEXT: vector.extract {{.*}}[1, 0]
  // AVX2-NEXT: vector.extract {{.*}}[2, 0]
  // AVX2-NEXT: vector.extract {{.*}}[3, 0]
  // AVX2-NEXT: vector.extract {{.*}}[4, 0]
  // AVX2-NEXT: vector.extract {{.*}}[5, 0]
  // AVX2-NEXT: vector.extract {{.*}}[6, 0]
  // AVX2-NEXT: vector.extract {{.*}}[7, 0]
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.insert {{.*}}[0]
  // AVX2-NEXT: vector.insert {{.*}}[1]
  // AVX2-NEXT: vector.insert {{.*}}[2]
  // AVX2-NEXT: vector.insert {{.*}}[3]
  // AVX2-NEXT: vector.insert {{.*}}[4]
  // AVX2-NEXT: vector.insert {{.*}}[5]
  // AVX2-NEXT: vector.insert {{.*}}[6]
  // AVX2-NEXT: vector.insert {{.*}}[7]
  // AVX2-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<8x1x8xf32>
  %0 = vector.transpose %arg0, [2, 1, 0] : vector<8x1x8xf32> to vector<8x1x8xf32>
  return %0 : vector<8x1x8xf32>
}

// -----

// AVX2-LABEL: func @transpose210_8x8x1
func @transpose210_8x8x1xf32(%arg0: vector<8x8x1xf32>) -> vector<1x8x8xf32> {
  //      AVX2: vector.shape_cast %{{.*}} : vector<8x8x1xf32> to vector<8x8xf32>
  // AVX2-NEXT: vector.extract {{.*}}[0]
  // AVX2-NEXT: vector.extract {{.*}}[1]
  // AVX2-NEXT: vector.extract {{.*}}[2]
  // AVX2-NEXT: vector.extract {{.*}}[3]
  // AVX2-NEXT: vector.extract {{.*}}[4]
  // AVX2-NEXT: vector.extract {{.*}}[5]
  // AVX2-NEXT: vector.extract {{.*}}[6]
  // AVX2-NEXT: vector.extract {{.*}}[7]
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.insert {{.*}}[0]
  // AVX2-NEXT: vector.insert {{.*}}[1]
  // AVX2-NEXT: vector.insert {{.*}}[2]
  // AVX2-NEXT: vector.insert {{.*}}[3]
  // AVX2-NEXT: vector.insert {{.*}}[4]
  // AVX2-NEXT: vector.insert {{.*}}[5]
  // AVX2-NEXT: vector.insert {{.*}}[6]
  // AVX2-NEXT: vector.insert {{.*}}[7]
  // AVX2-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<1x8x8xf32>
  %0 = vector.transpose %arg0, [2, 1, 0] : vector<8x8x1xf32> to vector<1x8x8xf32>
  return %0 : vector<1x8x8xf32>
}

// -----

// AVX2-LABEL: func @transpose210_1x8x8
func @transpose210_1x8x8xf32(%arg0: vector<1x8x8xf32>) -> vector<8x8x1xf32> {
  //      AVX2: vector.extract {{.*}}[0, 0]
  // AVX2-NEXT: vector.extract {{.*}}[0, 1]
  // AVX2-NEXT: vector.extract {{.*}}[0, 2]
  // AVX2-NEXT: vector.extract {{.*}}[0, 3]
  // AVX2-NEXT: vector.extract {{.*}}[0, 4]
  // AVX2-NEXT: vector.extract {{.*}}[0, 5]
  // AVX2-NEXT: vector.extract {{.*}}[0, 6]
  // AVX2-NEXT: vector.extract {{.*}}[0, 7]
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.shuffle {{.*}} [2, 10, 3, 11, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [2, 3, 8, 9, 6, 7, 12, 13] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0xcc", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-NEXT: llvm.inline_asm asm_dialect = intel "vblendps $0, $1, $2, 0x33", "=x,x,x" {{.*}} : (vector<8xf32>, vector<8xf32>) -> vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xf32>, vector<8xf32>
  // AVX2-COUNT-4: vector.shuffle {{.*}} [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xf32>, vector<8xf32>
  // AVX2-NEXT: vector.insert {{.*}}[0]
  // AVX2-NEXT: vector.insert {{.*}}[1]
  // AVX2-NEXT: vector.insert {{.*}}[2]
  // AVX2-NEXT: vector.insert {{.*}}[3]
  // AVX2-NEXT: vector.insert {{.*}}[4]
  // AVX2-NEXT: vector.insert {{.*}}[5]
  // AVX2-NEXT: vector.insert {{.*}}[6]
  // AVX2-NEXT: vector.insert {{.*}}[7]
  // AVX2-NEXT: vector.shape_cast %{{.*}} : vector<8x8xf32> to vector<8x8x1xf32>
  %0 = vector.transpose %arg0, [2, 1, 0] : vector<1x8x8xf32> to vector<8x8x1xf32>
  return %0 : vector<8x8x1xf32>
}

// -----

func @do_not_lower_nonf32_to_avx2(%arg0: vector<4x8xi32>) -> vector<8x4xi32> {
  %0 = vector.transpose %arg0, [1, 0] : vector<4x8xi32> to vector<8x4xi32>
  return %0 : vector<8x4xi32>
}

// AVX2-NOT: vector.shuffle

// -----

// AVX2-LABEL: func @transpose021_8x1x8
func @transpose021_8x1x8xf32(%arg0: vector<8x1x8xf32>) -> vector<8x8x1xf32> {
  %0 = vector.transpose %arg0, [0, 2, 1] : vector<8x1x8xf32> to vector<8x8x1xf32>
  return %0 : vector<8x8x1xf32>
}

// AVX2-NOT: vector.shuffle

// -----

// AVX2-LABEL: func @transpose021_8x8x1
func @transpose021_8x8x1xf32(%arg0: vector<8x8x1xf32>) -> vector<8x1x8xf32> {
  %0 = vector.transpose %arg0, [0, 2, 1] : vector<8x8x1xf32> to vector<8x1x8xf32>
  return %0 : vector<8x1x8xf32>
}

// AVX2-NOT: vector.shuffle

// -----

// ELTWISE-LABEL: func @transpose102_1x8x8xf32
// AVX2-LABEL: func @transpose102_1x8x8
func @transpose102_1x8x8xf32(%arg0: vector<1x8x8xf32>) -> vector<8x1x8xf32> {
  //      ELTWISE: vector.extract {{.*}}[0, 0] : vector<1x8x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [0, 0] : vector<8xf32> into vector<8x1x8xf32>
  // ELTWISE-NEXT: vector.extract {{.*}}[0, 1] : vector<1x8x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [1, 0] : vector<8xf32> into vector<8x1x8xf32>
  // ELTWISE-NEXT: vector.extract {{.*}}[0, 2] : vector<1x8x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [2, 0] : vector<8xf32> into vector<8x1x8xf32>
  // ELTWISE-NEXT: vector.extract {{.*}}[0, 3] : vector<1x8x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [3, 0] : vector<8xf32> into vector<8x1x8xf32>
  // ELTWISE-NEXT: vector.extract {{.*}}[0, 4] : vector<1x8x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [4, 0] : vector<8xf32> into vector<8x1x8xf32>
  // ELTWISE-NEXT: vector.extract {{.*}}[0, 5] : vector<1x8x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [5, 0] : vector<8xf32> into vector<8x1x8xf32>
  // ELTWISE-NEXT: vector.extract {{.*}}[0, 6] : vector<1x8x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [6, 0] : vector<8xf32> into vector<8x1x8xf32>
  // ELTWISE-NEXT: vector.extract {{.*}}[0, 7] : vector<1x8x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [7, 0] : vector<8xf32> into vector<8x1x8xf32>
  %0 = vector.transpose %arg0, [1, 0, 2] : vector<1x8x8xf32> to vector<8x1x8xf32>
  return %0 : vector<8x1x8xf32>
}

// AVX2-NOT: vector.shuffle

// -----

// ELTWISE-LABEL: func @transpose102_8x1x8xf32
// AVX2-LABEL: func @transpose102_8x1x8
func @transpose102_8x1x8xf32(%arg0: vector<8x1x8xf32>) -> vector<1x8x8xf32> {
  //      ELTWISE: vector.extract {{.*}}[0, 0] : vector<8x1x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [0, 0] : vector<8xf32> into vector<1x8x8xf32>
  // ELTWISE-NEXT: vector.extract {{.*}}[1, 0] : vector<8x1x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [0, 1] : vector<8xf32> into vector<1x8x8xf32>
  // ELTWISE-NEXT: vector.extract {{.*}}[2, 0] : vector<8x1x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [0, 2] : vector<8xf32> into vector<1x8x8xf32>
  // ELTWISE-NEXT: vector.extract {{.*}}[3, 0] : vector<8x1x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [0, 3] : vector<8xf32> into vector<1x8x8xf32>
  // ELTWISE-NEXT: vector.extract {{.*}}[4, 0] : vector<8x1x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [0, 4] : vector<8xf32> into vector<1x8x8xf32>
  // ELTWISE-NEXT: vector.extract {{.*}}[5, 0] : vector<8x1x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [0, 5] : vector<8xf32> into vector<1x8x8xf32>
  // ELTWISE-NEXT: vector.extract {{.*}}[6, 0] : vector<8x1x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [0, 6] : vector<8xf32> into vector<1x8x8xf32>
  // ELTWISE-NEXT: vector.extract {{.*}}[7, 0] : vector<8x1x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [0, 7] : vector<8xf32> into vector<1x8x8xf32>
  %0 = vector.transpose %arg0, [1, 0, 2] : vector<8x1x8xf32> to vector<1x8x8xf32>
  return %0 : vector<1x8x8xf32>
}

// AVX2-NOT: vector.shuffle

// -----

// ELTWISE-LABEL:   func @transpose1023_1x1x8x8xf32(
// AVX2-LABEL: func @transpose1023_1x1x8x8
func @transpose1023_1x1x8x8xf32(%arg0: vector<1x1x8x8xf32>) -> vector<1x1x8x8xf32> {
  // Note the single 2-D extract/insert pair since 2 and 3 are not transposed!
  //      ELTWISE: vector.extract {{.*}}[0, 0] : vector<1x1x8x8xf32>
  // ELTWISE-NEXT: vector.insert {{.*}} [0, 0] : vector<8x8xf32> into vector<1x1x8x8xf32>
  %0 = vector.transpose %arg0, [1, 0, 2, 3] : vector<1x1x8x8xf32> to vector<1x1x8x8xf32>
  return %0 : vector<1x1x8x8xf32>
}

// AVX2-NOT: vector.shuffle

// -----

// AVX2-LABEL: func @transpose120_1x8x8
func @transpose120_1x8x8xf32(%arg0: vector<1x8x8xf32>) -> vector<8x8x1xf32> {

  %0 = vector.transpose %arg0, [1, 2, 0] : vector<1x8x8xf32> to vector<8x8x1xf32>
  return %0 : vector<8x8x1xf32>
}

// AVX2-NOT: vector.shuffle

// -----

// AVX2-LABEL: func @transpose201_8x8x1
func @transpose201_8x8x1xf32(%arg0: vector<8x8x1xf32>) -> vector<1x8x8xf32> {
  %0 = vector.transpose %arg0, [2, 0, 1] : vector<8x8x1xf32> to vector<1x8x8xf32>
  return %0 : vector<1x8x8xf32>
}

// AVX2-NOT: vector.shuffle

