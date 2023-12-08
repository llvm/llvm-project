# RUN: llvm-mc %s -triple=riscv32 -mattr=+c,+d -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+c,+d -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c,+d < %s \
# RUN:     | llvm-objdump --mattr=+c,+d -M no-aliases -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+c,+d < %s \
# RUN:     | llvm-objdump --mattr=+c,+d -M no-aliases -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND %s

c.fld f8, (x9)
# CHECK-EXPAND: c.fsd fs0, 0(s1)
c.fsd f8, (x9)
# CHECK-EXPAND: c.fldsp fs0, 0(sp)
c.fldsp f8, (x2)
# CHECK-EXPAND: c.fsdsp fs0, 0(sp)
c.fsdsp f8, (x2)
