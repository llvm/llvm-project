// RUN: mlir-opt --test-emulate-narrow-int="memref-load-bitwidth=32" --cse --verify-diagnostics --split-input-file %s

func.func @transfer_read_non_subbyte_element(
    %arg0: memref<4x?x16xf16>, %arg1: index, %arg2: index, %arg3: index) {
  %cst = arith.constant 3.000000e+00 : f16
  // expected-error @below {{failed to legalize operation 'vector.transfer_read' that was explicitly marked illegal}}
  vector.transfer_read %arg0[%arg1, %arg2, %arg3], %cst :
    memref<4x?x16xf16>, vector<8xf16>
  return
}
