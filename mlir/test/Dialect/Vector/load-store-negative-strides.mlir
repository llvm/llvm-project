// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func.func @load_negative_stride(%base: memref<100x100xf32>) -> vector<8xf32> {
  // expected-error @+5 {{'vector.load' op memref strides must be non-negative}}
  %flip = memref.reinterpret_cast %base to
      offset: [0], sizes: [100, 100], strides: [-100, 1]
      : memref<100x100xf32> to memref<100x100xf32, strided<[-100, 1]>>
  %c0 = arith.constant 0 : index
  %v = vector.load %flip[%c0, %c0] : memref<100x100xf32, strided<[-100, 1]>>, vector<8xf32>
  return %v : vector<8xf32>
}

// -----

func.func @store_negative_stride(%base: memref<100x100xf32>, %val: vector<4xf32>) {
  // expected-error @+5 {{'vector.store' op memref strides must be non-negative}}
  %flip = memref.reinterpret_cast %base to
      offset: [0], sizes: [100, 100], strides: [-100, 1]
      : memref<100x100xf32> to memref<100x100xf32, strided<[-100, 1]>>
  %c0 = arith.constant 0 : index
  vector.store %val, %flip[%c0, %c0] : memref<100x100xf32, strided<[-100, 1]>>, vector<4xf32>
  return
}
