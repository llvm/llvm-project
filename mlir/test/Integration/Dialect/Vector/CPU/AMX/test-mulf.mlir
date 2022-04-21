// RUN: mlir-opt %s -convert-vector-to-scf -lower-affine -convert-scf-to-cf -convert-vector-to-llvm="enable-amx" -convert-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-translate -mlir-to-llvmir | \
// RUN: %lli --entry-function=entry --mattr="+amx-tile,+amx-int8,+amx-bf16" --dlopen=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// Note: To run this test, your CPU must support AMX.

// Multiply into zeroed destination.
func.func @kernel1(%arg0: memref<2x4xbf16>,
              %arg1: memref<2x4xbf16>,
              %arg2: memref<2x2xf32>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_load %arg0[%0, %0] : memref<2x4xbf16>  into vector<2x4xbf16>
  %2 = amx.tile_load %arg1[%0, %0] : memref<2x4xbf16>  into vector<2x4xbf16>
  %3 = amx.tile_zero : vector<2x2xf32>
  %4 = amx.tile_mulf %1, %2, %3 : vector<2x4xbf16>, vector<2x4xbf16>, vector<2x2xf32>
  amx.tile_store %arg2[%0, %0], %4 : memref<2x2xf32>, vector<2x2xf32>
  return
}

// Multiply and update into destination.
func.func @kernel2(%arg0: memref<2x4xbf16>,
              %arg1: memref<2x4xbf16>,
              %arg2: memref<2x2xf32>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_load %arg0[%0, %0] : memref<2x4xbf16>  into vector<2x4xbf16>
  %2 = amx.tile_load %arg1[%0, %0] : memref<2x4xbf16>  into vector<2x4xbf16>
  %3 = amx.tile_load %arg2[%0, %0] : memref<2x2xf32> into vector<2x2xf32>
  %4 = amx.tile_mulf %1, %2, %3 : vector<2x4xbf16>, vector<2x4xbf16>, vector<2x2xf32>
  amx.tile_store %arg2[%0, %0], %4 : memref<2x2xf32>, vector<2x2xf32>
  return
}

func.func @entry() -> i32 {
  %f0 = arith.constant 0.0: f32
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index

  // Set up memory.
  %a = memref.alloc() : memref<2x4xbf16>
  %b = memref.alloc() : memref<2x4xbf16>
  %c = memref.alloc() : memref<2x2xf32>

  %0 = arith.constant dense<[[1.0, 2.0, 3.0, 4.0 ],
                          [5.0, 6.0, 7.0, 8.0 ]]> : vector<2x4xbf16>
  vector.transfer_write %0, %a[%c0, %c0] : vector<2x4xbf16>, memref<2x4xbf16>
  %1 = arith.constant dense<[[ 9.0, 10.0, 11.0, 12.0 ],
                          [13.0, 14.0, 15.0, 16.0 ]]> : vector<2x4xbf16>
  vector.transfer_write %1, %b[%c0, %c0] : vector<2x4xbf16>, memref<2x4xbf16>

  // Call kernel.
  call @kernel1(%a, %b, %c) : (memref<2x4xbf16>, memref<2x4xbf16>, memref<2x2xf32>) -> ()

  // Print and verify.
  //
  // CHECK:      ( 124, 144 )
  // CHECK-NEXT: ( 308, 360 )
  scf.for %i = %c0 to %c2 step %c1 {
    %av = vector.transfer_read %c[%i, %c0], %f0: memref<2x2xf32>, vector<2xf32>
    vector.print %av : vector<2xf32>
  }

  // Call kernel.
  call @kernel2(%a, %b, %c) : (memref<2x4xbf16>, memref<2x4xbf16>, memref<2x2xf32>) -> ()

  // Print and verify.
  //
  // CHECK-NEXT: ( 248, 288 )
  // CHECK-NEXT: ( 616, 720 )
  //
  scf.for %i = %c0 to %c2 step %c1 {
    %cv = vector.transfer_read %c[%i, %c0], %f0: memref<2x2xf32>, vector<2xf32>
    vector.print %cv : vector<2xf32>
  }

  // Release resources.
  memref.dealloc %a : memref<2x4xbf16>
  memref.dealloc %b : memref<2x4xbf16>
  memref.dealloc %c : memref<2x2xf32>

  %i0 = arith.constant 0 : i32
  return %i0 : i32
}
