// RUN: mlir-opt %s -convert-vector-to-scf -lower-affine -convert-scf-to-cf -convert-vector-to-llvm="enable-amx" -convert-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-translate -mlir-to-llvmir | \
// RUN: %lli --entry-function=entry --mattr="+amx-tile,+amx-int8,+amx-bf16" --dlopen=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// Note: To run this test, your CPU must support AMX.

func.func @tilezero(%arg0: memref<?x?xi32>, %i: index, %j: index) {
  %1 = amx.tile_zero : vector<16x16xi32>
  amx.tile_store %arg0[%i, %j], %1 : memref<?x?xi32>, vector<16x16xi32>
  return
}

func.func @entry() -> i32 {
  %i0 = arith.constant 0: i32
  %i1 = arith.constant 1: i32
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c3 = arith.constant 3: index
  %c19 = arith.constant 19: index

  // Set up memory.
  %a = memref.alloc(%c19, %c19) : memref<?x?xi32>
  scf.for %i = %c0 to %c19 step %c1 {
    scf.for %j = %c0 to %c19 step %c1 {
      memref.store %i1, %a[%i, %j] : memref<?x?xi32>
    }
  }

  // Call kernel.
  call @tilezero(%a, %c1, %c1) : (memref<?x?xi32>, index, index) -> ()

  // Print and verify that the tilezero is correctly strided within
  // the enveloping 19x19 buffer.
  //
  // CHECK:      ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
  // CHECK-NEXT: ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
  //
  scf.for %i = %c0 to %c19 step %c1 {
    %av = vector.transfer_read %a[%i, %c0], %i0: memref<?x?xi32>, vector<19xi32>
    vector.print %av : vector<19xi32>
  }

  // Call kernel with different indices.
  call @tilezero(%a, %c0, %c3) : (memref<?x?xi32>, index, index) -> ()

  // Print and verify that the tilezero is again correctly strided
  // within the enveloping 19x19 buffer.
  //
  // CHECK-NEXT: ( 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 )
  // CHECK-NEXT: ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
  // CHECK-NEXT: ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
  //
  scf.for %i = %c0 to %c19 step %c1 {
    %av = vector.transfer_read %a[%i, %c0], %i0: memref<?x?xi32>, vector<19xi32>
    vector.print %av : vector<19xi32>
  }

  // Release resources.
  memref.dealloc %a : memref<?x?xi32>

  return %i0 : i32
}
