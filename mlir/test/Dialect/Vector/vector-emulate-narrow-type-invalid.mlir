// RUN: mlir-opt --test-emulate-narrow-int="arith-compute-bitwidth=1 memref-load-bitwidth=8" --verify-diagnostics --split-input-file %s

// Masked reads are not supported, the mask must not be dropped.

func.func @negative_transfer_read_vector_mask(%idx: index, %mask: vector<8xi1>) -> vector<8xi4> {
  %pad = arith.constant 0 : i4
  %alloc = memref.alloc() : memref<8xi4>
  // expected-error @below {{failed to legalize operation 'vector.transfer_read' that was explicitly marked illegal}}
  %v = vector.mask %mask { vector.transfer_read %alloc[%idx], %pad : memref<8xi4>, vector<8xi4> } : vector<8xi1> -> vector<8xi4>
  return %v : vector<8xi4>
}
