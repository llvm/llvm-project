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
