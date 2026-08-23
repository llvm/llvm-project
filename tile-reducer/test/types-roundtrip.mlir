// RUN: tr-opt %s | tr-opt | FileCheck %s

// Milestone 2: tiles are static; buffers may be static, `?`, or named.

// CHECK-LABEL: func.func @tiles
func.func @tiles(%t1: !tr.tile<128xf32>, %t2: !tr.tile<128x128xf32>,
                 %t3: !tr.tile<64x32xi32>, %t0: !tr.tile<f32>) {
  // CHECK: !tr.tile<128xf32>
  // CHECK: !tr.tile<128x128xf32>
  // CHECK: !tr.tile<64x32xi32>
  // CHECK: !tr.tile<f32>
  return
}

// CHECK-LABEL: func.func @buffers
func.func @buffers(%a: !tr.buffer<MxKxf32>, %b: !tr.buffer<Mxf32>,
                   %c: !tr.buffer<?x?xf32>, %d: !tr.buffer<128x256xf16>) {
  // CHECK: !tr.buffer<MxKxf32>
  // CHECK: !tr.buffer<Mxf32>
  // CHECK: !tr.buffer<?x?xf32>
  // CHECK: !tr.buffer<128x256xf16>
  return
}
