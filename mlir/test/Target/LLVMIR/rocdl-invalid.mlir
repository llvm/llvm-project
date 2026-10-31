// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// expected-error @below {{rocdl.flat_work_group_size must match rocdl.reqd_work_group_size}}
llvm.func @reqd_work_group_size_flat_work_group_size_mismatch()
    attributes {rocdl.kernel,
      rocdl.flat_work_group_size = "16,128",
      rocdl.reqd_work_group_size = array<i32: 32, 2, 1>} {
  llvm.return
}

// -----

// expected-error @below {{rocdl.max_flat_work_group_size must match rocdl.reqd_work_group_size}}
llvm.func @reqd_work_group_size_max_flat_work_group_size_mismatch()
    attributes {rocdl.kernel,
      rocdl.max_flat_work_group_size = 128 : index,
      rocdl.reqd_work_group_size = array<i32: 32, 2, 1>} {
  llvm.return
}

// -----

llvm.func @rocdl_deprecated_ignore_denormal_mode(%ptr: !llvm.ptr<1>, %data: f32) {
  // expected-warning @below {{rocdl.ignore_denormal_mode is deprecated; use ignore_denormal_mode on llvm.atomicrmw instead}}
  llvm.atomicrmw fadd %ptr, %data monotonic {rocdl.ignore_denormal_mode} : !llvm.ptr<1>, f32
  llvm.return
}

// -----

llvm.func @mfma_invalid_cd_regclass(%a: f32, %b: f32, %c: vector<4xf32>) -> vector<4xf32> {
  // expected-error @below {{expected string or keyword containing one of the following enum values for attribute 'cd_regclass' [auto, require_agpr, require_vgpr]}}
  %d = rocdl.mfma.f32.16x16x4f32 %a, %b, %c, 0, 0, none cd_regclass = 3 : (f32, f32, vector<4xf32>) -> vector<4xf32>
  llvm.return %d : vector<4xf32>
}

// -----

llvm.func @mfma_scale_invalid_cd_regclass(%a: vector<8xi32>, %b: vector<8xi32>, %c: vector<4xf32>, %sa: i32, %sb: i32) -> vector<4xf32> {
  // expected-error @below {{expected string or keyword containing one of the following enum values for attribute 'cd_regclass' [auto, require_agpr, require_vgpr]}}
  %d = rocdl.mfma.scale.f32.16x16x128.f8f6f4 %a, %b, %c, fp8_e4m3, fp8_e4m3, 0, %sa, 0, %sb cd_regclass = -1 : (vector<8xi32>, vector<8xi32>, vector<4xf32>, i32, i32) -> vector<4xf32>
  llvm.return %d : vector<4xf32>
}
