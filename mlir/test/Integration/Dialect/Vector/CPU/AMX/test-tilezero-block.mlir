// RUN: mlir-opt %s -convert-vector-to-scf -lower-affine -convert-scf-to-cf -convert-vector-to-llvm="enable-amx" -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-translate -mlir-to-llvmir | \
// RUN: %lli --entry-function=entry --mattr="+amx-tile,+amx-int8,+amx-bf16" --dlopen=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// Note: To run this test, your CPU must support AMX.

func.func @print(%arg0: memref<4x32xf32>) {
  %fu = arith.constant -1.0: f32
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c4 = arith.constant 4: index
  scf.for %i = %c0 to %c4 step %c1 {
    %0 = vector.transfer_read %arg0[%i, %c0], %fu: memref<4x32xf32>, vector<32xf32>
    vector.print %0 : vector<32xf32>
  }
  return
}

func.func @kernel(%arg0: memref<4x32xf32>) {
  %c0  = arith.constant 0: index
  %c2  = arith.constant 2 : index
  %c4  = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  scf.for %i = %c0 to %c4 step %c2 {
    scf.for %j = %c0 to %c32 step %c16 {
      %0 = amx.tile_zero : vector<2x16xf32>
      amx.tile_store %arg0[%i, %j], %0 : memref<4x32xf32>, vector<2x16xf32>
      func.call @print(%arg0) : (memref<4x32xf32>) -> ()
    }
  }
  return
}

func.func @entry() -> i32 {
  %f1  = arith.constant 1.0: f32
  %c0  = arith.constant 0: index
  %c1  = arith.constant 1: index
  %c4  = arith.constant 4 : index
  %c32 = arith.constant 32 : index

  // Set up memory.
  %a = memref.alloc() : memref<4x32xf32>
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c32 step %c1 {
      memref.store %f1, %a[%i, %j] : memref<4x32xf32>
    }
  }

  // Call kernel.
  func.call @kernel(%a) : (memref<4x32xf32>) -> ()

  // Verify progress of blocked tilezero.
  //
  // CHECK:      ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
  // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
  // CHECK-NEXT: ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
  // CHECK-NEXT: ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
  //
  // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
  // CHECK-NEXT: ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
  //
  // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
  // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
  //
  // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
  //

  // Release resources.
  memref.dealloc %a : memref<4x32xf32>

  %i0 = arith.constant 0 : i32
  return %i0 : i32
}
