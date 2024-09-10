// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

// NOTE: This file tests lowerings that are implemented in the X86Vector
// dialect. Since X86 does not support scalable vectors, all examples in this
// file use fixed-width vectors.

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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_transpose avx2_lowering_strategy = true
    } : !transform.op<"func.func">
    transform.yield
  }
}
