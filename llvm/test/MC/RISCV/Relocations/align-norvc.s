## To ensure ALIGN relocations in norvc code can adapt to shrinking of preceding rvc code,
## we generate $alignment-2 bytes of NOPs regardless of rvc.
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s -o %t
# RUN: llvm-objdump -dr -M no-aliases %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax -riscv-align-rvc=0 %s -o %t0
# RUN: llvm-objdump -dr -M no-aliases %t0 | FileCheck %s --check-prefix=CHECK0

# CHECK:               00000000: R_RISCV_RELAX        *ABS*
# CHECK-NEXT:       4: 0001      <unknown>
# CHECK-NEXT:          00000004: R_RISCV_ALIGN        *ABS*+0x6
# CHECK-NEXT:       6: 00000013  addi zero, zero, 0x0
# CHECK-NEXT:       a: 00000537  lui a0, 0x0

# CHECK0:              00000000: R_RISCV_RELAX        *ABS*
# CHECK0-NEXT:      4: 00000013  addi zero, zero, 0x0
# CHECK0-NEXT:         00000004: R_RISCV_ALIGN        *ABS*+0x4
# CHECK0-NEXT:      8: 00000537  lui a0, 0x0

  lui a0, %hi(foo)
  .option norvc
.balign 8
  lui a0, %hi(foo)
