// RUN: mlir-opt %s --split-input-file --verify-diagnostics

func.func @element_type(%m: memref<8x8xf16>, %i: index) {
  // expected-error @below {{matrix and memref element types differ}}
  %a = air.simdgroup_matrix_load %m[%i, %i] : memref<8x8xf16> -> vector<64xf32>
  return
}

// -----

func.func @indices(%m: memref<8x8xf32>, %i: index) {
  // expected-error @below {{takes two indices}}
  %a = air.simdgroup_matrix_load %m[%i] : memref<8x8xf32> -> vector<64xf32>
  return
}
