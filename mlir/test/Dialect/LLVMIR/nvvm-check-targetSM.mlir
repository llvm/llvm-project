// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// Just check these don't emit errors.
gpu.module @check_valid_SM_exact [#nvvm.target<chip = "sm_80">] {
  test.nvvm_requires_sm_80
}

gpu.module @check_valid_SM_greater_1 [#nvvm.target<chip = "sm_86">] {
  test.nvvm_requires_sm_80
}

gpu.module @check_valid_SM_greater_2 [#nvvm.target<chip = "sm_90">] {
  test.nvvm_requires_sm_80
}

gpu.module @check_valid_SM_arch_acc_1 [#nvvm.target<chip = "sm_90a">] {
  test.nvvm_requires_sm_90a
}

gpu.module @check_valid_SM_arch_acc_2 [#nvvm.target<chip = "sm_90a">] {
  test.nvvm_requires_sm_80
}

gpu.module @check_valid_SM_arch_acc_multi_1 [#nvvm.target<chip = "sm_90a">] {
  test.nvvm_requires_sm_90a_or_sm_100a
}

gpu.module @check_valid_SM_arch_acc_multi_2 [#nvvm.target<chip = "sm_100a">] {
  test.nvvm_requires_sm_90a_or_sm_100a
}


gpu.module @disable_verify_target1 [#nvvm.target<chip = "sm_90", verifyTarget = false>] {
  test.nvvm_requires_sm_90a
}

gpu.module @disable_verify_target2 [#nvvm.target<chip = "sm_70", verifyTarget = false>] {
  test.nvvm_requires_sm_80
}

gpu.module @disable_verify_target3 [#nvvm.target<chip = "sm_90", verifyTarget = false>] {
  test.nvvm_requires_sm_90a_or_sm_100a
}

// -----

gpu.module @check_invalid_SM_lesser_1 [#nvvm.target<chip = "sm_70">] {
  // expected-error @below {{is not supported on sm_70}}
  test.nvvm_requires_sm_80
}

// -----

gpu.module @check_invalid_SM_lesser_2 [#nvvm.target<chip = "sm_75">] {
  // expected-error @below {{is not supported on sm_75}}
  test.nvvm_requires_sm_80
}

// -----

gpu.module @check_invalid_SM_arch_acc_1 [#nvvm.target<chip = "sm_90">] {
  // expected-error @below {{is not supported on sm_90}}
  test.nvvm_requires_sm_90a
}

// -----

gpu.module @check_invalid_SM_arch_acc_2 [#nvvm.target<chip = "sm_80">] {
  // expected-error @below {{is not supported on sm_80}}
  test.nvvm_requires_sm_90a
}

// -----

gpu.module @check_invalid_SM_arch_acc_multi_1 [#nvvm.target<chip = "sm_80">] {
  // expected-error @below {{is not supported on sm_80}}
  test.nvvm_requires_sm_90a_or_sm_100a
}

// -----

gpu.module @check_invalid_SM_arch_acc_multi_2 [#nvvm.target<chip = "sm_90">] {
  // expected-error @below {{is not supported on sm_90}}
  test.nvvm_requires_sm_90a_or_sm_100a
}
