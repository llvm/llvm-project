//RUN: llvm-mc  -triple=aarch64-arm-none-eabi -o - %s | FileCheck %s

// CHECK: .cfi_negate_ra_state_with_pc
foo:
  .cfi_startproc
  .cfi_negate_ra_state_with_pc
  .cfi_endproc
