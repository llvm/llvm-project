# RUN: llvm-mc -triple=riscv32 -mattr=+zclsd -M no-aliases < %s \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zclsd < %s \
# RUN:     | llvm-objdump --no-print-imm-hex --mattr=+zclsd -d -M no-aliases - \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND %s

# CHECK-EXPAND: c.ld s0, 0(s1)
c.ld x8, (x9)
# CHECK-EXPAND: c.sd s0, 0(s1)
c.sd x8, (x9)
# CHECK-EXPAND: c.ldsp s0, 0(sp)
c.ldsp x8, (x2)
# CHECK-EXPAND: c.sdsp s0, 0(sp)
c.sdsp x8, (x2)
# CHECK-EXPAND: c.ldsp s2, 0(sp)
c.ldsp x18, (x2)
# CHECK-EXPAND: c.sdsp s2, 0(sp)
c.sdsp x18, (x2)
# CHECK-EXPAND: c.sdsp zero, 0(sp)
c.sdsp x0, (x2)
