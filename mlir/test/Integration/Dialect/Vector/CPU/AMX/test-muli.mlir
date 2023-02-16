// RUN: mlir-opt %s -convert-vector-to-scf -lower-affine -convert-scf-to-cf -convert-vector-to-llvm="enable-amx" -finalize-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-translate -mlir-to-llvmir | \
// RUN: %lli --entry-function=entry --mattr="+amx-tile,+amx-int8,+amx-bf16" --dlopen=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// Note: To run this test, your CPU must support AMX.

// Multiply into zeroed destination.
func.func @kernel1(%arg0: memref<2x8xi8>,
              %arg1: memref<2x8xi8>,
              %arg2: memref<2x2xi32>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_load %arg0[%0, %0] : memref<2x8xi8>  into vector<2x8xi8>
  %2 = amx.tile_load %arg1[%0, %0] : memref<2x8xi8>  into vector<2x8xi8>
  %3 = amx.tile_zero : vector<2x2xi32>
  %4 = amx.tile_muli %1 zext, %2 zext, %3 : vector<2x8xi8>, vector<2x8xi8>, vector<2x2xi32>
  amx.tile_store %arg2[%0, %0], %4 : memref<2x2xi32>, vector<2x2xi32>
  return
}

// Multiply and update into destination.
func.func @kernel2(%arg0: memref<2x8xi8>,
              %arg1: memref<2x8xi8>,
              %arg2: memref<2x2xi32>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_load %arg0[%0, %0] : memref<2x8xi8>  into vector<2x8xi8>
  %2 = amx.tile_load %arg1[%0, %0] : memref<2x8xi8>  into vector<2x8xi8>
  %3 = amx.tile_load %arg2[%0, %0] : memref<2x2xi32> into vector<2x2xi32>
  %4 = amx.tile_muli %1 zext, %2 zext, %3 : vector<2x8xi8>, vector<2x8xi8>, vector<2x2xi32>
  amx.tile_store %arg2[%0, %0], %4 : memref<2x2xi32>, vector<2x2xi32>
  return
}

func.func @entry() -> i32 {
  %i0 = arith.constant 0: i32
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index

  // Set up memory.
  %a = memref.alloc() : memref<2x8xi8>
  %b = memref.alloc() : memref<2x8xi8>
  %c = memref.alloc() : memref<2x2xi32>

  %0 = arith.constant dense<[[1 , 2,  3 , 4 , 5,  6,  7,  8],
                           [9, 10, 11, 12, 13, 14, 15, 16]]> : vector<2x8xi8>
  vector.transfer_write %0, %a[%c0, %c0] : vector<2x8xi8>, memref<2x8xi8>
  %1 = arith.constant dense<[[17, 18, 19, 20, 21, 22, 23, 24],
                           [25, 26, 27, 28, 29, 30, 31, 32]]> : vector<2x8xi8>
  vector.transfer_write %1, %b[%c0, %c0] : vector<2x8xi8>, memref<2x8xi8>

  // Call kernel.
  call @kernel1(%a, %b, %c) : (memref<2x8xi8>, memref<2x8xi8>, memref<2x2xi32>) -> ()

  // Print and verify.
  //
  // CHECK:      ( 884, 1028 )
  // CHECK-NEXT: ( 2324, 2724 )
  scf.for %i = %c0 to %c2 step %c1 {
    %av = vector.transfer_read %c[%i, %c0], %i0: memref<2x2xi32>, vector<2xi32>
    vector.print %av : vector<2xi32>
  }

  // Call kernel.
  call @kernel2(%a, %b, %c) : (memref<2x8xi8>, memref<2x8xi8>, memref<2x2xi32>) -> ()

  // Print and verify.
  //
  // CHECK-NEXT: ( 1768, 2056 )
  // CHECK-NEXT: ( 4648, 5448 )
  //
  scf.for %i = %c0 to %c2 step %c1 {
    %cv = vector.transfer_read %c[%i, %c0], %i0: memref<2x2xi32>, vector<2xi32>
    vector.print %cv : vector<2xi32>
  }

  // Release resources.
  memref.dealloc %a : memref<2x8xi8>
  memref.dealloc %b : memref<2x8xi8>
  memref.dealloc %c : memref<2x2xi32>

  return %i0 : i32
}
