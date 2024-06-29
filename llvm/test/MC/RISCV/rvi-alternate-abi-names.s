# RUN: llvm-mc %s -triple=riscv32 \
# RUN:     | FileCheck %s
# RUN: llvm-mc %s -triple=riscv64 \
# RUN:     | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r - \
# RUN:     | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r - \
# RUN:     | FileCheck %s

# 'fp' is an alternate ABI name for 's0' and it should be accepted in input.
# However, 's0' should be printed in preference.

# CHECK: addi s0, s0, -4
addi fp, fp, -4
