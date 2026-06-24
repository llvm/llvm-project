// RUN: mlir-opt %s --print-buffer-lifetime-stats --split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: --- Buffer Lifetime Statistics for 'sequential_non_overlapping' ---
// CHECK:   Tracked allocations     : 2
// CHECK:   Total allocated bytes    : 8192
// CHECK:   Peak live bytes          : 4096
// CHECK:   Non-overlapping pairs    : 1
// CHECK:   Buffer: memref<1024xf32> | size=4096 | lifetime=[1, 3)
// CHECK:   Buffer: memref<512xf64> | size=4096 | lifetime=[5, 7)
// CHECK: ---

func.func @sequential_non_overlapping(%arg0: memref<1024xf32>,
                                       %arg1: memref<512xf64>) {
  %cst = arith.constant 0.0 : f32
  %a = memref.alloc() : memref<1024xf32>       // 1024 * 4 = 4096 bytes
  linalg.fill ins(%cst : f32) outs(%a : memref<1024xf32>)
  memref.dealloc %a : memref<1024xf32>

  %cst2 = arith.constant 0.0 : f64
  %b = memref.alloc() : memref<512xf64>        // 512 * 8 = 4096 bytes
  linalg.fill ins(%cst2 : f64) outs(%b : memref<512xf64>)
  memref.dealloc %b : memref<512xf64>
  return
}

// -----

// CHECK-LABEL: --- Buffer Lifetime Statistics for 'overlapping_lifetimes' ---
// CHECK:   Tracked allocations     : 2
// CHECK:   Total allocated bytes    : 6144
// CHECK:   Peak live bytes          : 6144
// CHECK:   Non-overlapping pairs    : 0
// CHECK:   Buffer: memref<512xf32> | size=2048 | lifetime=[0, 4)
// CHECK:   Buffer: memref<1024xf32> | size=4096 | lifetime=[1, 3)
// CHECK: ---

func.func @overlapping_lifetimes() {
  %a = memref.alloc() : memref<512xf32>        // 2048 bytes
  %b = memref.alloc() : memref<1024xf32>       // 4096 bytes
  %cst = arith.constant 0.0 : f32
  memref.dealloc %b : memref<1024xf32>
  memref.dealloc %a : memref<512xf32>
  return
}

// -----

// CHECK-LABEL: --- Buffer Lifetime Statistics for 'three_buffers_mixed' ---
// CHECK:   Tracked allocations     : 3
// CHECK:   Total allocated bytes    : 10240
// CHECK:   Peak live bytes          : 8192
// CHECK:   Non-overlapping pairs    : 1
// CHECK:   Buffer: memref<512xf32> | size=2048 | lifetime=[0, 2)
// CHECK:   Buffer: memref<1024xf32> | size=4096 | lifetime=[1, 5)
// CHECK:   Buffer: memref<1024xf32> | size=4096 | lifetime=[3, 6)
// CHECK: ---

// %a and %b overlap (a=[0,2), b=[1,5))
// %a and %c don't overlap (a=[0,2), c=[3,6))
// %b and %c overlap (b=[1,5), c=[3,6))
// So 1 non-overlapping pair: (%a, %c)
func.func @three_buffers_mixed() {
  %a = memref.alloc() : memref<512xf32>        // 2048 bytes
  %b = memref.alloc() : memref<1024xf32>       // 4096 bytes
  memref.dealloc %a : memref<512xf32>
  %c = memref.alloc() : memref<1024xf32>       // 4096 bytes
  %cst = arith.constant 0.0 : f32
  memref.dealloc %b : memref<1024xf32>
  memref.dealloc %c : memref<1024xf32>
  return
}

// -----

// CHECK-LABEL: --- Buffer Lifetime Statistics for 'single_alloc' ---
// CHECK:   Tracked allocations     : 1
// CHECK:   Total allocated bytes    : 256
// CHECK:   Peak live bytes          : 256
// CHECK:   Non-overlapping pairs    : 0
// CHECK:   Buffer: memref<64xf32> | size=256 | lifetime=[0, 1)
// CHECK: ---

func.func @single_alloc() {
  %a = memref.alloc() : memref<64xf32>         // 64 * 4 = 256 bytes
  memref.dealloc %a : memref<64xf32>
  return
}

// -----

// CHECK-LABEL: --- Buffer Lifetime Statistics for 'no_allocs' ---
// CHECK:   Tracked allocations     : 0
// CHECK:   Total allocated bytes    : 0
// CHECK:   Peak live bytes          : 0
// CHECK:   Non-overlapping pairs    : 0
// CHECK: ---

func.func @no_allocs(%arg0: memref<1024xf32>) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<1024xf32>)
  return
}
