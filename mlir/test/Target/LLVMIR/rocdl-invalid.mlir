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
