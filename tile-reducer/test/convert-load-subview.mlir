// RUN: tr-opt %s --convert-tr-buffers-to-memref --convert-tr-to-linalg | FileCheck %s

// Milestone 9: tr.load is a subview of the input buffer. The 128x128 tile
// is not allocated. The accumulator is a 128-element alloca.

// CHECK-LABEL: func.func @row_sum
// CHECK-SAME: (%[[IN:.*]]: memref<?x?xf32>, %[[OUT:.*]]: memref<?xf32>)
func.func @row_sum(%in: !tr.buffer<MxKxf32>, %out: !tr.buffer<Mxf32>) {
  %row_blk     = tr.program_id 0 : index
  %c128        = arith.constant 128 : index
  %k           = tr.dim %in, 1 : !tr.buffer<MxKxf32>, index
  %num_k_tiles = arith.divui %k, %c128 : index

  // CHECK: %[[ACC:.*]] = memref.alloca() : memref<128xf32>
  // CHECK: linalg.fill
  %zero = tr.constant 0.0 : !tr.tile<128xf32>

  // CHECK: tr.for
  %result = tr.for %kt = 0 to %num_k_tiles step 1
      iter_args(%acc = %zero) -> !tr.tile<128xf32> {
    // Tile coordinates * tile size become subview offsets.
    // CHECK: %[[ROFF:.*]] = arith.muli %{{.*}}, %{{.*}} : index
    // CHECK: %[[COFF:.*]] = arith.muli %{{.*}}, %{{.*}} : index
    // CHECK: %[[VIEW:.*]] = memref.subview %[[IN]][%[[ROFF]], %[[COFF]]] [128, 128] [1, 1]
    // CHECK-NOT: memref.alloc(
    // CHECK-NOT: memref.alloca() : memref<128x128
    %t       = tr.load %in[%row_blk, %kt]
        : !tr.buffer<MxKxf32>, !tr.tile<128x128xf32>

    // CHECK: linalg.generic
    // CHECK-SAME: iterator_types = ["parallel", "reduction"]
    // CHECK-SAME: ins(%[[VIEW]] :
    %partial = tr.reduce_sum %t, axis = 1
        : !tr.tile<128x128xf32> -> !tr.tile<128xf32>

    %acc2    = tr.add %acc, %partial
        : !tr.tile<128xf32>

    tr.yield %acc2 : !tr.tile<128xf32>
  }

  // CHECK: memref.subview %[[OUT]]
  // CHECK: memref.copy
  tr.store %out[%row_blk], %result : !tr.buffer<Mxf32>, !tr.tile<128xf32>
  return
}
