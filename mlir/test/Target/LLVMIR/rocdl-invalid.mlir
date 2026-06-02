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
