// RUN: mlir-opt %s -convert-vector-to-gpu \
// RUN: | mlir-opt -gpu-lower-to-nvvm-pipeline="cubin-format=%gpu_compilation_format" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s
// Check that arith.subf and arith.negf applied to the result of a matmul are
// lowered through gpu.subgroup_mma_elementwise to NVVM. With A[i][j] = j and
// C[i][j] = i, the kernel computes -(C - (A * A + C)) = A * A, whose rows are
// all [-0, 120, 240, ...] (the first column is the negation of 0).
#map_a = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_c = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @main() {
  %0 = memref.alloc() : memref<16x16xf16>
  %22 = memref.alloc() : memref<16x16xf16>
  %1 = memref.alloc() : memref<16x16xf32>

  %f0 = arith.constant 0.0e+00 : f16
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index

  // Initialize the input matrix with the column index and the accumulator
  // with the row index.
  scf.for %arg0 = %c0 to %c16 step %c1 {
    scf.for %arg1 = %c0 to %c16 step %c1 {
      %2 = arith.index_cast %arg1 : index to i16
      %3 = arith.sitofp %2 : i16 to f16
      memref.store %3, %0[%arg0, %arg1] : memref<16x16xf16>
      %4 = arith.index_cast %arg0 : index to i16
      %5 = arith.sitofp %4 : i16 to f16
      memref.store %5, %22[%arg0, %arg1] : memref<16x16xf16>
    }
  }

  %3 = memref.cast %1 : memref<16x16xf32> to memref<*xf32>

  // Copy the input and the accumulator to the device.
  %token = gpu.wait async
  %d0, %t0 = gpu.alloc async [%token] () : memref<16x16xf16>
  %d22, %t1 = gpu.alloc async [%token] () : memref<16x16xf16>
  %x = gpu.memcpy async [%token] %d0, %0 : memref<16x16xf16>, memref<16x16xf16>
  %y = gpu.memcpy async [%token] %d22, %22 : memref<16x16xf16>, memref<16x16xf16>
  gpu.wait [%token]

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c32, %block_y = %c1, %block_z = %c1) {
    %A = vector.transfer_read %d0[%c0, %c0], %f0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %B = vector.transfer_read %d0[%c0, %c0], %f0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %C = vector.transfer_read %d22[%c0, %c0], %f0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %D = vector.contract {indexing_maps = [#map_a, #map_b, #map_c], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
    %E = arith.subf %C, %D : vector<16x16xf16>
    %F = arith.negf %E : vector<16x16xf16>
    vector.transfer_write %F, %d22[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf16>, memref<16x16xf16>
    gpu.terminator
  }

  // Copy the result back to the host.
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
  // CHECK-COUNT-16: [-0, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440, 1560, 1680, 1800]
  return
}

func.func private @printMemrefF32(memref<*xf32>)
