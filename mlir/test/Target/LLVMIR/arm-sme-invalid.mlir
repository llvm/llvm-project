// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// Verify shape of predicate and vector must match
llvm.func @arm_sme_vector_to_tile_invalid_types(%tileslice : i32,
                                                %nxv4i1 : vector<[4]xi1>,
                                                %nxv16i8 : vector<[16]xi8>) {
  %tile = llvm.mlir.constant(0 : index) : i32
  // expected-error @+1 {{failed to verify that all of {pg, vector} have same shape}}
  "arm_sme.intr.write.horiz"(%tile, %tileslice, %nxv4i1, %nxv16i8) :
      (i32, i32, vector<[4]xi1>, vector<[16]xi8>) -> ()
  llvm.return
}

// -----

llvm.func @arm_sme_tile_slice_to_vector_invalid_shapes(
  %tileslice : i32, %nxv4i1 : vector<[4]xi1>, %nxv16i8 : vector<[16]xi8>
) -> vector<[3]xf32> {
  %tile = llvm.mlir.constant(0 : index) : i32
  // expected-error @+1 {{failed to verify that all of {vector, pg, res} have same shape}}
  %res = "arm_sme.intr.read.horiz"(%nxv16i8, %nxv4i1, %tile, %tileslice) :
      (vector<[16]xi8>, vector<[4]xi1>, i32, i32) -> vector<[3]xf32>
  llvm.return %res : vector<[3]xf32>
}

// -----

llvm.func @arm_sme_tile_slice_to_vector_invalid_element_types(
  %tileslice : i32, %nxv4i1 : vector<[4]xi1>, %nxv4f32 : vector<[4]xf32>
) -> vector<[3]xi32> {
  %tile = llvm.mlir.constant(0 : index) : i32
  // expected-error @+1 {{failed to verify that all of {vector, res} have same element type}}
  %res = "arm_sme.intr.read.horiz"(%nxv4f32, %nxv4i1, %tile, %tileslice) :
      (vector<[4]xf32>, vector<[4]xi1>, i32, i32) -> vector<[4]xi32>
  llvm.return %res : vector<[4]xi32>
}
