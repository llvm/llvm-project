// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts |\
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:   -shared-libs=%mlir_c_runner_utils
// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm -finalize-memref-to-llvm='use-aligned-alloc=1' -convert-func-to-llvm -arith-expand -reconcile-unrealized-casts |\
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:   -shared-libs=%mlir_c_runner_utils | FileCheck %s

// FIXME: Windows does not have aligned_alloc
// UNSUPPORTED: system-windows

func.func @entry() {
  // Set up memory.
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c8 = arith.constant 8: index
  %A = memref.alloc() : memref<8xf32>
  scf.for %i = %c0 to %c8 step %c1 {
    %i32 = arith.index_cast %i : index to i32
    %fi = arith.sitofp %i32 : i32 to f32
    memref.store %fi, %A[%i] : memref<8xf32>
  }

  %d0 = arith.constant -1.0 : f32
  %Av = vector.transfer_read %A[%c0], %d0: memref<8xf32>, vector<8xf32>
  vector.print %Av : vector<8xf32>
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7 )

  // Realloc with static sizes.
  %B = memref.realloc %A : memref<8xf32> to memref<10xf32>

  %c10 = arith.constant 10: index
  scf.for %i = %c8 to %c10 step %c1 {
    %i32 = arith.index_cast %i : index to i32
    %fi = arith.sitofp %i32 : i32 to f32
    memref.store %fi, %B[%i] : memref<10xf32>
  }

  %Bv = vector.transfer_read %B[%c0], %d0: memref<10xf32>, vector<10xf32>
  vector.print %Bv : vector<10xf32>
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 )

  // Realloc with dynamic sizes.
  %Bd = memref.cast %B : memref<10xf32> to memref<?xf32>
  %c13 = arith.constant 13: index
  %Cd = memref.realloc %Bd(%c13) : memref<?xf32> to memref<?xf32>
  %C = memref.cast %Cd : memref<?xf32> to memref<13xf32>

  scf.for %i = %c10 to %c13 step %c1 {
    %i32 = arith.index_cast %i : index to i32
    %fi = arith.sitofp %i32 : i32 to f32
    memref.store %fi, %C[%i] : memref<13xf32>
  }

  %Cv = vector.transfer_read %C[%c0], %d0: memref<13xf32>, vector<13xf32>
  vector.print %Cv : vector<13xf32>
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 )

  memref.dealloc %C : memref<13xf32>
  return
}
