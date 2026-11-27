// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=8" | FileCheck %s

// Verify the masks of the vectorized reads and writes: each one enables what is
// left of the memref from the transfer index.

// CHECK-LABEL: func.func @shifted_read
// CHECK:         affine.for %[[IV:.*]] = 0 to 63 step 8
// CHECK:           %[[IDX:.*]] = affine.apply #{{.*}}(%[[IV]])
// CHECK:           %[[SIZE:.*]] = arith.subi %{{.*}}, %[[IDX]] : index
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[SIZE]] : vector<8xi1>
// CHECK:           vector.mask %[[MASK]] { vector.transfer_read %{{.*}}[%[[IDX]]]
func.func @shifted_read(%src: memref<64xf32>, %dst: memref<64xf32>) {
  affine.for %i = 0 to 63 {
    %v = affine.load %src[%i + 1] : memref<64xf32>
    affine.store %v, %dst[%i] : memref<64xf32>
  }
  return
}

// CHECK-LABEL: func.func @shifted_write
// CHECK:         affine.for %[[IV:.*]] = 0 to 60 step 8
// CHECK:           %[[IDX:.*]] = affine.apply #{{.*}}(%[[IV]])
// CHECK:           %[[SIZE:.*]] = arith.subi %{{.*}}, %[[IDX]] : index
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[SIZE]] : vector<8xi1>
// CHECK:           vector.mask %[[MASK]] { vector.transfer_write %{{.*}}, %{{.*}}[%[[IDX]]]
func.func @shifted_write(%src: memref<64xf32>, %dst: memref<64xf32>) {
  affine.for %i = 0 to 60 {
    %v = affine.load %src[%i] : memref<64xf32>
    affine.store %v, %dst[%i + 4] : memref<64xf32>
  }
  return
}

// The inner loop is vectorized: only the memref dim it runs over is masked.

// CHECK-LABEL: func.func @memref_2d_vector_1d
// CHECK:         affine.for %[[I:.*]] = 0 to 7
// CHECK:           affine.for %[[J:.*]] = 0 to 100 step 8
// CHECK:             %[[C100:.*]] = arith.constant 100 : index
// CHECK:             %[[RSIZE:.*]] = arith.subi %[[C100]], %[[J]] : index
// CHECK:             %[[RMASK:.*]] = vector.create_mask %[[RSIZE]] : vector<8xi1>
// CHECK:             vector.mask %[[RMASK]] { vector.transfer_read %{{.*}}[%[[I]], %[[J]]]
// CHECK:             %[[C100_1:.*]] = arith.constant 100 : index
// CHECK:             %[[WSIZE:.*]] = arith.subi %[[C100_1]], %[[J]] : index
// CHECK:             %[[WMASK:.*]] = vector.create_mask %[[WSIZE]] : vector<8xi1>
// CHECK:             vector.mask %[[WMASK]] { vector.transfer_write %{{.*}}, %{{.*}}[%[[I]], %[[J]]]
func.func @memref_2d_vector_1d(%src: memref<7x100xf32>, %dst: memref<7x100xf32>) {
  affine.for %i = 0 to 7 {
    affine.for %j = 0 to 100 {
      %v = affine.load %src[%i, %j] : memref<7x100xf32>
      affine.store %v, %dst[%i, %j] : memref<7x100xf32>
    }
  }
  return
}
