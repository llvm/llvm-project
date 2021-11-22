// RUN: mlir-opt %s -test-vector-transpose-lowering=eltwise=1 | FileCheck %s --check-prefix=ELTWISE
// RUN: mlir-opt %s -test-vector-transpose-lowering=shuffle=1 | FileCheck %s --check-prefix=SHUFFLE
// RUN: mlir-opt %s -test-vector-transpose-lowering=flat=1 | FileCheck %s --check-prefix=FLAT
// RUN: mlir-opt %s -test-vector-transpose-lowering=avx2=1 | FileCheck %s --check-prefix=AVX2

// ELTWISE-LABEL: func @transpose23
// ELTWISE-SAME: %[[A:.*]]: vector<2x3xf32>
// ELTWISE:      %[[Z:.*]] = arith.constant dense<0.000000e+00> : vector<3x2xf32>
// ELTWISE:      %[[T0:.*]] = vector.extract %[[A]][0, 0] : vector<2x3xf32>
// ELTWISE:      %[[T1:.*]] = vector.insert %[[T0]], %[[Z]] [0, 0] : f32 into vector<3x2xf32>
// ELTWISE:      %[[T2:.*]] = vector.extract %[[A]][1, 0] : vector<2x3xf32>
// ELTWISE:      %[[T3:.*]] = vector.insert %[[T2]], %[[T1]] [0, 1] : f32 into vector<3x2xf32>
// ELTWISE:      %[[T4:.*]] = vector.extract %[[A]][0, 1] : vector<2x3xf32>
// ELTWISE:      %[[T5:.*]] = vector.insert %[[T4]], %[[T3]] [1, 0] : f32 into vector<3x2xf32>
// ELTWISE:      %[[T6:.*]] = vector.extract %[[A]][1, 1] : vector<2x3xf32>
// ELTWISE:      %[[T7:.*]] = vector.insert %[[T6]], %[[T5]] [1, 1] : f32 into vector<3x2xf32>
// ELTWISE:      %[[T8:.*]] = vector.extract %[[A]][0, 2] : vector<2x3xf32>
// ELTWISE:      %[[T9:.*]] = vector.insert %[[T8]], %[[T7]] [2, 0] : f32 into vector<3x2xf32>
// ELTWISE:      %[[T10:.*]] = vector.extract %[[A]][1, 2] : vector<2x3xf32>
// ELTWISE:      %[[T11:.*]] = vector.insert %[[T10]], %[[T9]] [2, 1] : f32 into vector<3x2xf32>
// ELTWISE:      return %[[T11]] : vector<3x2xf32>
func @transpose23(%arg0: vector<2x3xf32>) -> vector<3x2xf32> {
  %0 = vector.transpose %arg0, [1, 0] : vector<2x3xf32> to vector<3x2xf32>
  return %0 : vector<3x2xf32>
}

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

// AVX2-LABEL: func @transpose8x8
func @transpose8x8xf32(%arg0: vector<8x8xf32>) -> vector<8x8xf32> {
  //      AVX2: vector.shuffle {{.*}} [0, 8, 1, 9, 4, 12, 5, 13] : vector<8xf32>, vector<8xf32>
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
  %0 = vector.transpose %arg0, [1, 0] : vector<8x8xf32> to vector<8x8xf32>
  return %0 : vector<8x8xf32>
}
