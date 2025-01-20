// RUN: mlir-opt %s --mlir-very-unsafe-disable-verifier-on-parsing  -verify-diagnostics

// Just check these don't emit errors.

gpu.module @check_invalid_disabled_SM_lesser_1 [#nvvm.target<chip = "sm_70">] {
  test.nvvm_requires_sm_80
}

gpu.module @check_invalid_disabled_SM_lesser_2 [#nvvm.target<chip = "sm_75">] {
  test.nvvm_requires_sm_80
}

gpu.module @check_invalid_disabled_SM_arch_acc_1 [#nvvm.target<chip = "sm_90">] {
  test.nvvm_requires_sm_90a
}

gpu.module @check_invalid_disabled_SM_arch_acc_2 [#nvvm.target<chip = "sm_80">] {
  test.nvvm_requires_sm_90a
}
