// Note: borrowed/copied from mlir/test/Target/LLVMIR/arm-sme-invalid.mlir

// Check that verify-diagnostics=only-expected passes with only one actual `expected-error`
// RUN: mlir-translate %s -verify-diagnostics=only-expected -split-input-file -mlir-to-llvmir

// Check that verify-diagnostics=all fails because we're missing three `expected-error`
// RUN: not mlir-translate %s -verify-diagnostics=all -split-input-file -mlir-to-llvmir 2>&1 | FileCheck %s --check-prefix=CHECK-VERIFY-ALL
// CHECK-VERIFY-ALL:      error: unexpected error: 'arm_sme.intr.write.horiz' op failed to verify that all of {predicate, vector} have same shape
// CHECK-VERIFY-ALL-NEXT: "arm_sme.intr.write.horiz"
// CHECK-VERIFY-ALL:      error: unexpected error: 'arm_sme.intr.read.horiz' op failed to verify that all of {vector, res} have same element type
// CHECK-VERIFY-ALL-NEXT: %res = "arm_sme.intr.read.horiz"
// CHECK-VERIFY-ALL:      error: unexpected error: 'arm_sme.intr.cntsb' op failed to verify that `res` is i64
// CHECK-VERIFY-ALL-NEXT: %res = "arm_sme.intr.cntsb"

llvm.func @arm_sme_vector_to_tile_invalid_types(%tileslice : i32,
                                                %nxv4i1 : vector<[4]xi1>,
                                                %nxv16i8 : vector<[16]xi8>) {
  "arm_sme.intr.write.horiz"(%tileslice, %nxv4i1, %nxv16i8) <{tile_id = 0 : i32}> :
      (i32, vector<[4]xi1>, vector<[16]xi8>) -> ()
  llvm.return
}

// -----

llvm.func @arm_sme_tile_slice_to_vector_invalid_shapes(
  %tileslice : i32, %nxv4i1 : vector<[4]xi1>, %nxv16i8 : vector<[16]xi8>
) -> vector<[3]xf32> {
  // expected-error @+1 {{failed to verify that all of {vector, predicate, res} have same shape}}
  %res = "arm_sme.intr.read.horiz"(%nxv16i8, %nxv4i1, %tileslice) <{tile_id = 0 : i32}> :
      (vector<[16]xi8>, vector<[4]xi1>, i32) -> vector<[3]xf32>
  llvm.return %res : vector<[3]xf32>
}

// -----

llvm.func @arm_sme_tile_slice_to_vector_invalid_element_types(
  %tileslice : i32, %nxv4i1 : vector<[4]xi1>, %nxv4f32 : vector<[4]xf32>
) -> vector<[3]xi32> {
  %res = "arm_sme.intr.read.horiz"(%nxv4f32, %nxv4i1, %tileslice) <{tile_id = 0 : i32}> :
      (vector<[4]xf32>, vector<[4]xi1>, i32) -> vector<[4]xi32>
  llvm.return %res : vector<[4]xi32>
}

// -----

llvm.func @arm_sme_streaming_vl_invalid_return_type() -> i32 {
  %res = "arm_sme.intr.cntsb"() : () -> i32
  llvm.return %res : i32
}
