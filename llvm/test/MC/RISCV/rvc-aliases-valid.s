# RUN: llvm-mc %s -triple=riscv32 -mattr=+c -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+c -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -M no-aliases -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+c < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -M no-aliases -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND %s

# CHECK-EXPAND: c.lw s0, 0(s1)
c.lw x8, (x9)
# CHECK-EXPAND: c.sw s0, 0(s1)
c.sw x8, (x9)
# CHECK-EXPAND: c.lwsp s0, 0(sp)
c.lwsp x8, (x2)
# CHECK-EXPAND: c.swsp s0, 0(sp)
c.swsp x8, (x2)
