# RUN: llvm-mc %s -triple=riscv32 -mattr=+zilsd -M no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zilsd < %s \
# RUN:     | llvm-objdump --no-print-imm-hex --mattr=+zilsd -M no-aliases -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND %s

# CHECK-EXPAND: ld a0, 0(a1)
ld x10, (x11)
# CHECK-EXPAND: sd a0, 0(a1)
sd x10, (x11)
