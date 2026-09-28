// RUN: mlir-opt %s -convert-vector-to-gpu \
// RUN: | mlir-opt -gpu-lower-to-nvvm-pipeline="cubin-format=%gpu_compilation_format" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s
// Check that a transposed vector.transfer_write lowered to a transposed
// gpu.subgroup_mma_store_matrix stores the transpose of twice the input.
func.func @main() {
  %0 = memref.alloc() : memref<16x16xf16>
  %22 = memref.alloc() : memref<16x16xf16>
  %1 = memref.alloc() : memref<16x16xf32>

  %f0 = arith.constant 0.0e+00 : f16
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index

  // Initialize the input matrix with 16 * row + column.
  scf.for %arg0 = %c0 to %c16 step %c1 {
    scf.for %arg1 = %c0 to %c16 step %c1 {
      %2 = arith.muli %arg0, %c16 : index
      %3 = arith.addi %2, %arg1 : index
      %4 = arith.index_cast %3 : index to i16
      %5 = arith.sitofp %4 : i16 to f16
      memref.store %5, %0[%arg0, %arg1] : memref<16x16xf16>
    }
  }
  // Initialize the output matrix with zeros.
  scf.for %arg0 = %c0 to %c16 step %c1 {
    scf.for %arg1 = %c0 to %c16 step %c1 {
      memref.store %f0, %22[%arg0, %arg1] : memref<16x16xf16>
    }
  }

  %3 = memref.cast %1 : memref<16x16xf32> to memref<*xf32>

  // Copy the input and the output matrices to the device.
  %token = gpu.wait async
  %d0, %t0 = gpu.alloc async [%token] () : memref<16x16xf16>
  %d22, %t1 = gpu.alloc async [%token] () : memref<16x16xf16>
  %x = gpu.memcpy async [%token] %d0, %0 : memref<16x16xf16>, memref<16x16xf16>
  %y = gpu.memcpy async [%token] %d22, %22 : memref<16x16xf16>, memref<16x16xf16>
  gpu.wait [%token]

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c32, %block_y = %c1, %block_z = %c1) {
    %A = vector.transfer_read %d0[%c0, %c0], %f0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %B = arith.addf %A, %A : vector<16x16xf16>
    vector.transfer_write %B, %d22[%c0, %c0] {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : vector<16x16xf16>, memref<16x16xf16>
    gpu.terminator
  }

  // Copy the output matrix back to the host.
  %token2 = gpu.wait async
  %z = gpu.memcpy async [%token2] %22, %d22 : memref<16x16xf16>, memref<16x16xf16>
  %w = gpu.dealloc async [%token2] %d0 : memref<16x16xf16>
  %v = gpu.dealloc async [%token2] %d22 : memref<16x16xf16>
  gpu.wait [%token2]

  // Convert the results from f16 to f32 for printing.
  scf.for %arg0 = %c0 to %c16 step %c1 {
    scf.for %arg1 = %c0 to %c16 step %c1 {
      %6 = memref.load %22[%arg0, %arg1] : memref<16x16xf16>
      %7 = arith.extf %6 : f16 to f32
      memref.store %7, %1[%arg0, %arg1] : memref<16x16xf32>
    }
  }

  // Print the memref after computation.
  call @printMemrefF32(%3) : (memref<*xf32>) -> ()
  // CHECK:      [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480],
  // CHECK-NEXT: [2, 34, 66, 98, 130, 162, 194, 226, 258, 290, 322, 354, 386, 418, 450, 482],
  // CHECK-NEXT: [4, 36, 68, 100, 132, 164, 196, 228, 260, 292, 324, 356, 388, 420, 452, 484],
  // CHECK-NEXT: [6, 38, 70, 102, 134, 166, 198, 230, 262, 294, 326, 358, 390, 422, 454, 486],
  // CHECK-NEXT: [8, 40, 72, 104, 136, 168, 200, 232, 264, 296, 328, 360, 392, 424, 456, 488],
  // CHECK-NEXT: [10, 42, 74, 106, 138, 170, 202, 234, 266, 298, 330, 362, 394, 426, 458, 490],
  // CHECK-NEXT: [12, 44, 76, 108, 140, 172, 204, 236, 268, 300, 332, 364, 396, 428, 460, 492],
  // CHECK-NEXT: [14, 46, 78, 110, 142, 174, 206, 238, 270, 302, 334, 366, 398, 430, 462, 494],
  // CHECK-NEXT: [16, 48, 80, 112, 144, 176, 208, 240, 272, 304, 336, 368, 400, 432, 464, 496],
  // CHECK-NEXT: [18, 50, 82, 114, 146, 178, 210, 242, 274, 306, 338, 370, 402, 434, 466, 498],
  // CHECK-NEXT: [20, 52, 84, 116, 148, 180, 212, 244, 276, 308, 340, 372, 404, 436, 468, 500],
  // CHECK-NEXT: [22, 54, 86, 118, 150, 182, 214, 246, 278, 310, 342, 374, 406, 438, 470, 502],
  // CHECK-NEXT: [24, 56, 88, 120, 152, 184, 216, 248, 280, 312, 344, 376, 408, 440, 472, 504],
  // CHECK-NEXT: [26, 58, 90, 122, 154, 186, 218, 250, 282, 314, 346, 378, 410, 442, 474, 506],
  // CHECK-NEXT: [28, 60, 92, 124, 156, 188, 220, 252, 284, 316, 348, 380, 412, 444, 476, 508],
  // CHECK-NEXT: [30, 62, 94, 126, 158, 190, 222, 254, 286, 318, 350, 382, 414, 446, 478, 510]]
  return
}

func.func private @printMemrefF32(memref<*xf32>)
