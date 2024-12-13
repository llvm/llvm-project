# RUN: llvm-mc -triple riscv32 %s \
# RUN:   | FileCheck -check-prefix=CHECK-ASM %s
# RUN: llvm-mc -triple riscv64 %s \
# RUN:   | FileCheck -check-prefix=CHECK-ASM %s

# RUN: llvm-mc -filetype=obj -triple riscv32 %s \
# RUN:   | llvm-objdump -dr -M no-aliases - \
# RUN:   | FileCheck -check-prefix=CHECK-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv64 %s \
# RUN:   | llvm-objdump -dr -M no-aliases  - \
# RUN:   | FileCheck -check-prefix=CHECK-OBJ %s

  # CHECK-ASM: .text
  # CHECK-OBJ: <.text>:

  nop
  # CHECK-ASM: nop
  # CHECK-OBJ: addi zero, zero, 0x0

  .reloc ., R_RISCV_VENDOR,    VENDOR_NAME
  .reloc ., R_RISCV_CUSTOM192, my_foo + 1
  addi a0, a0, 0
  # CHECK-ASM: [[L1:.L[^:]+]]:
  # CHECK-ASM-NEXT: .reloc [[L1]], R_RISCV_VENDOR, VENDOR_NAME
  # CHECK-ASM-NEXT: [[L2:.L[^:]+]]:
  # CHECK-ASM-NEXT: .reloc [[L2]], R_RISCV_CUSTOM192, my_foo+1
  # CHECK-ASM-NEXT: mv a0, a0

  # CHECK-OBJ: addi a0, a0, 0
  # CHECK-OBJ-NEXT: R_RISCV_VENDOR    VENDOR_NAME
  # CHECK-OBJ-NEXT: R_RISCV_CUSTOM192 my_foo+0x1

  nop
  # CHECK-ASM: nop
  # CHECK-OBJ: addi zero, zero, 0x0
