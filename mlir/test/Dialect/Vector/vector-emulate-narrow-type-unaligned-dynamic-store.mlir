// RUN: mlir-opt --test-emulate-narrow-int="arith-compute-bitwidth=1 memref-load-bitwidth=8" --verify-diagnostics --split-input-file %s

// Dynamic sub-byte vector.store offsets cannot be treated as byte-aligned unless
// the caller explicitly opts into the alignment contract.
func.func @vector_store_i2_2d_dynamic_col(%arg0: vector<4xi2>, %idx0: index,
                                          %idx1: index) {
  %src = memref.alloc() : memref<3x4xi2>
  // expected-error @below {{failed to legalize operation 'vector.store' that was explicitly marked illegal}}
  vector.store %arg0, %src[%idx0, %idx1] : memref<3x4xi2>, vector<4xi2>
  return
}

// -----

func.func @vector_store_i4_dynamic_memref(%arg0: vector<8xi4>, %dim0: index,
                                          %dim1: index, %idx0: index,
                                          %idx1: index) {
  %src = memref.alloc(%dim0, %dim1) : memref<?x?xi4>
  // expected-error @below {{failed to legalize operation 'vector.store' that was explicitly marked illegal}}
  vector.store %arg0, %src[%idx0, %idx1] : memref<?x?xi4>, vector<8xi4>
  return
}
