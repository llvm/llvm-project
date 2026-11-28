// RUN: split-file %s %t
// RUN: mlir-opt --test-emulate-narrow-int="arith-compute-bitwidth=1 memref-load-bitwidth=8" --verify-diagnostics --split-input-file %t/masked-read.mlir
// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=32" --cse --verify-diagnostics --split-input-file %t/non-subbyte-read.mlir

//--- masked-read.mlir

// Masked reads are not supported, the mask must not be dropped.

func.func @negative_transfer_read_vector_mask(%idx: index, %mask: vector<8xi1>) -> vector<8xi4> {
  %pad = arith.constant 0 : i4
  %alloc = memref.alloc() : memref<8xi4>
  // expected-error @below {{failed to legalize operation 'vector.transfer_read' that was explicitly marked illegal}}
  %v = vector.mask %mask { vector.transfer_read %alloc[%idx], %pad : memref<8xi4>, vector<8xi4> } : vector<8xi1> -> vector<8xi4>
  return %v : vector<8xi4>
}

//--- non-subbyte-read.mlir

func.func @transfer_read_non_subbyte_element(
    %arg0: memref<4x?x16xf16>, %arg1: index, %arg2: index, %arg3: index) {
  %cst = arith.constant 3.000000e+00 : f16
  // expected-error @below {{failed to legalize operation 'vector.transfer_read' that was explicitly marked illegal}}
  vector.transfer_read %arg0[%arg1, %arg2, %arg3], %cst :
    memref<4x?x16xf16>, vector<8xf16>
  return
}
