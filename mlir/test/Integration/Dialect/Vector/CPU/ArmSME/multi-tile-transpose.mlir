// RUN: mlir-opt %s -test-lower-to-arm-sme -test-lower-to-llvm | \
// RUN: %mcr_aarch64_cmd \
// RUN:   -e=main -entry-point-result=void \
// RUN:   -march=aarch64 -mattr="+sve,+sme" \
// RUN:   -shared-libs=%native_mlir_runner_utils,%native_mlir_c_runner_utils,%native_arm_sme_abi_shlib,%native_mlir_arm_runner_utils | \
// RUN: FileCheck %s

#transpose = affine_map<(d0, d1) -> (d1, d0)>

func.func @fill2DMemrefRows(%memref: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %rows = memref.dim %memref, %c0 : memref<?x?xf32>
  %cols = memref.dim %memref, %c1 : memref<?x?xf32>
  scf.for %i = %c0 to %rows step %c1 {
    scf.for %j = %c0 to %cols step %c1 {
      %n = arith.addi %i, %c1 : index
      %val = arith.index_cast %n : index to i32
      %val_f32 = arith.sitofp %val : i32 to f32
      memref.store %val_f32, %memref[%i, %j] : memref<?x?xf32>
    }
  }
  return
}

func.func @testTransposedReadWithMask(%maskRows: index, %maskCols: index) {
  %in = memref.alloca() : memref<4x16xf32>
  %out = memref.alloca() : memref<16x4xf32>

  %inDyn = memref.cast %in : memref<4x16xf32> to memref<?x?xf32>
  %outDyn = memref.cast %out : memref<16x4xf32> to memref<?x?xf32>

  func.call @fill2DMemrefRows(%inDyn) : (memref<?x?xf32>) -> ()

  /// A mask so we only read the first maskRows x maskCols portion of %in.
  %mask = vector.create_mask %maskRows, %maskCols : vector<[4]x[16]xi1>
  %pad = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index

  /// A vector.transfer_read with a transpose permutation map. So the input data
  /// (and mask) have a [4]x[16] shape, but the output is [16]x[4].
  %readTransposed = vector.transfer_read %inDyn[%c0, %c0], %pad, %mask
    {permutation_map = #transpose, in_bounds = [true, true]} : memref<?x?xf32>, vector<[16]x[4]xf32>

  /// Write the vector back to a memref (that also has a transposed shape).
  vector.transfer_write %readTransposed, %outDyn[%c0, %c0] {in_bounds = [true, true]} : vector<[16]x[4]xf32>, memref<?x?xf32>

  /// Print the input memref.
  vector.print str "Input memref:\n"
  %inUnranked = memref.cast %inDyn : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%inUnranked) : (memref<*xf32>) -> ()

  /// Print the result memref.
  vector.print str "Masked transposed result:\n"
  %outUnranked = memref.cast %outDyn : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%outUnranked) : (memref<*xf32>) -> ()

  return
}

func.func @testTransposedWriteWithMask(%maskRows: index, %maskCols: index) {
  %in = memref.alloca() : memref<16x4xf32>
  %out = memref.alloca() : memref<4x16xf32>

  %c0_f32 = arith.constant 0.0 : f32
  linalg.fill ins(%c0_f32 : f32) outs(%out : memref<4x16xf32>)

  %inDyn = memref.cast %in : memref<16x4xf32> to memref<?x?xf32>
  %outDyn = memref.cast %out : memref<4x16xf32> to memref<?x?xf32>

  func.call @fill2DMemrefRows(%inDyn) : (memref<?x?xf32>) -> ()

  /// A regular read.
  %c0 = arith.constant 0 : index
  %read = vector.transfer_read %inDyn[%c0, %c0], %c0_f32 {in_bounds = [true, true]}
    : memref<?x?xf32>, vector<[16]x[4]xf32>

  /// A mask so we only write the first maskRows x maskCols portion of transpose(%in).
  %mask = vector.create_mask %maskRows, %maskCols : vector<[4]x[16]xi1>

  /// Write out the data with a transpose. Here (like the read test) the mask
  /// matches the shape of the memory, not the vector.
  vector.transfer_write %read, %outDyn[%c0, %c0], %mask {permutation_map = #transpose, in_bounds = [true, true]}
    : vector<[16]x[4]xf32>, memref<?x?xf32>

  /// Print the input memref.
  vector.print str "Input memref:\n"
  %inUnranked = memref.cast %inDyn : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%inUnranked) : (memref<*xf32>) -> ()

  /// Print the result memref.
  vector.print str "Masked transposed result:\n"
  %outUnranked = memref.cast %outDyn : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%outUnranked) : (memref<*xf32>) -> ()

  return
}

func.func @main() {
  /// Set the SVL to 128-bits (i.e. vscale = 1).
  /// This test is for the use of multiple tiles rather than scalability.
  %c128 = arith.constant 128 : i32
  func.call @setArmSVLBits(%c128) : (i32) -> ()

  //      CHECK: Input memref:
  //      CHECK:  [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]
  // CHECK-NEXT:  [2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2]
  // CHECK-NEXT:  [3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3,   3]
  // CHECK-NEXT:  [4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4]
  //
  //      CHECK:  Masked transposed result:
  //      CHECK:  [1,   2,   0,   0]
  // CHECK-NEXT:  [1,   2,   0,   0]
  // CHECK-NEXT:  [1,   2,   0,   0]
  // CHECK-NEXT:  [1,   2,   0,   0]
  // CHECK-NEXT:  [1,   2,   0,   0]
  // CHECK-NEXT:  [1,   2,   0,   0]
  // CHECK-NEXT:  [1,   2,   0,   0]
  // CHECK-NEXT:  [1,   2,   0,   0]
  // CHECK-NEXT:  [1,   2,   0,   0]
  // CHECK-NEXT:  [1,   2,   0,   0]
  // CHECK-NEXT:  [1,   2,   0,   0]
  // CHECK-NEXT:  [1,   2,   0,   0]
  // CHECK-NEXT:  [1,   2,   0,   0]
  // CHECK-NEXT:  [1,   2,   0,   0]
  // CHECK-NEXT:  [1,   2,   0,   0]
  // CHECK-NEXT:  [0,   0,   0,   0]
  %readMaskRows = arith.constant 2 : index
  %readMaskCols = arith.constant 15 : index
  func.call @testTransposedReadWithMask(%readMaskRows, %readMaskCols) : (index, index) -> ()

  //      CHECK: Input memref:
  //      CHECK:  [1,   1,   1,   1]
  // CHECK-NEXT:  [2,   2,   2,   2]
  // CHECK-NEXT:  [3,   3,   3,   3]
  // CHECK-NEXT:  [4,   4,   4,   4]
  // CHECK-NEXT:  [5,   5,   5,   5]
  // CHECK-NEXT:  [6,   6,   6,   6]
  // CHECK-NEXT:  [7,   7,   7,   7]
  // CHECK-NEXT:  [8,   8,   8,   8]
  // CHECK-NEXT:  [9,   9,   9,   9]
  // CHECK-NEXT:  [10,   10,   10,   10]
  // CHECK-NEXT:  [11,   11,   11,   11]
  // CHECK-NEXT:  [12,   12,   12,   12]
  // CHECK-NEXT:  [13,   13,   13,   13]
  // CHECK-NEXT:  [14,   14,   14,   14]
  // CHECK-NEXT:  [15,   15,   15,   15]
  // CHECK-NEXT:  [16,   16,   16,   16]
  //
  //      CHECK:  Masked transposed result:
  //      CHECK:  [1,   2,   3,   4,   5,   6,   7,   8,   0,   0,   0,   0,   0,   0,   0,   0]
  // CHECK-NEXT:  [1,   2,   3,   4,   5,   6,   7,   8,   0,   0,   0,   0,   0,   0,   0,   0]
  // CHECK-NEXT:  [1,   2,   3,   4,   5,   6,   7,   8,   0,   0,   0,   0,   0,   0,   0,   0]
  // CHECK-NEXT:  [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
  %writeMaskRows = arith.constant 3 : index
  %writeMaskCols = arith.constant 8 : index
  func.call @testTransposedWriteWithMask(%writeMaskRows, %writeMaskCols) : (index, index) -> ()

  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
func.func private @setArmSVLBits(%bits : i32)
