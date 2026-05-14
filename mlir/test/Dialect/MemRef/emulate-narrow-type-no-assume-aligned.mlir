// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=8" --cse --verify-diagnostics --split-input-file %s

// Without `assume-aligned=true`, dynamic offsets in `memref.subview` and
// `memref.reinterpret_cast` cannot be proven to be multiples of
// `dstBits / srcBits`. The patterns must reject them so partial conversion
// fails to legalize the op.

func.func @negative_subview_dynamic_inner_offset_i4(%off: index) -> i4 {
  %c0 = arith.constant 0 : index
  %arr = memref.alloc() : memref<128xi4>
  // expected-error @+1 {{failed to legalize operation 'memref.subview' that was explicitly marked illegal}}
  %subview = memref.subview %arr[%off] [32] [1] : memref<128xi4> to memref<32xi4, strided<[1], offset: ?>>
  %ld = memref.load %subview[%c0] : memref<32xi4, strided<[1], offset: ?>>
  return %ld : i4
}

// -----

func.func @negative_reinterpret_cast_memref_rank3_dynamic_offset_i4(%arg0: memref<2x4x8xi4>, %off: index) -> memref<4x4x8xi4, strided<[32, 8, 1], offset: ?>> {
  // expected-error @+1 {{failed to legalize operation 'memref.reinterpret_cast' that was explicitly marked illegal}}
  %r = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [4, 4, 8], strides: [32, 8, 1] : memref<2x4x8xi4> to memref<4x4x8xi4, strided<[32, 8, 1], offset: ?>>
  return %r : memref<4x4x8xi4, strided<[32, 8, 1], offset: ?>>
}
