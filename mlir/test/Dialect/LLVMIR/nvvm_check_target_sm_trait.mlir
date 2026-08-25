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

gpu.module @check_valid_SM_family_1 [#nvvm.target<chip = "sm_100f">] {
  test.nvvm_requires_sm_100f
}

gpu.module @check_valid_SM_family_2 [#nvvm.target<chip = "sm_100a">] {
  test.nvvm_requires_sm_100f
}

gpu.module @check_valid_SM_family_3 [#nvvm.target<chip = "sm_103a">] {
  test.nvvm_requires_sm_100f
}

gpu.module @check_valid_SM_family_4[#nvvm.target<chip = "sm_103f">] {
  test.nvvm_requires_sm_100f
}

gpu.module @check_valid_SM_family_multi_1 [#nvvm.target<chip = "sm_100f">] {
  test.nvvm_requires_sm_100f_or_sm_120f
}

gpu.module @check_valid_SM_family_multi_2 [#nvvm.target<chip = "sm_120f">] {
  test.nvvm_requires_sm_100f_or_sm_120f
}

gpu.module @check_valid_SM_arch_or_family_1 [#nvvm.target<chip = "sm_90a">] {
  test.nvvm_requires_sm_90a_or_sm_100f
}

gpu.module @check_valid_SM_arch_or_family_2 [#nvvm.target<chip = "sm_100f">] {
  test.nvvm_requires_sm_90a_or_sm_100f
}

gpu.module @check_valid_SM_arch_or_family_3 [#nvvm.target<chip = "sm_103a">] {
  test.nvvm_requires_sm_90a_or_sm_100f
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

gpu.module @disable_verify_target4 [#nvvm.target<chip = "sm_120f", verifyTarget = false>] {
  test.nvvm_requires_sm_100f
}

gpu.module @disable_verify_target5 [#nvvm.target<chip = "sm_100", verifyTarget = false>] {
  test.nvvm_requires_sm_100f_or_sm_120f
}

gpu.module @disable_verify_target6 [#nvvm.target<chip = "sm_90", verifyTarget = false>] {
  test.nvvm_requires_sm_90a_or_sm_100f
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

// -----

gpu.module @check_invalid_SM_family [#nvvm.target<chip = "sm_110a">] {
  // expected-error @below {{is not supported on sm_110a}}
  test.nvvm_requires_sm_100f
}

// -----

gpu.module @check_invalid_SM_family_higher [#nvvm.target<chip = "sm_100f">] {
  // expected-error @below {{is not supported on sm_100f}}
  test.nvvm_requires_sm_103f
}

// -----

gpu.module @check_invalid_SM_family_multi [#nvvm.target<chip = "sm_110a">] {
  // expected-error @below {{is not supported on sm_110a}}
  test.nvvm_requires_sm_100f_or_sm_120f
}

// -----

gpu.module @check_invalid_SM_arch_or_family [#nvvm.target<chip = "sm_100">] {
  // expected-error @below {{is not supported on sm_100}}
  test.nvvm_requires_sm_90a_or_sm_100f
}
