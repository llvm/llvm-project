# RUN: llvm-mc %s -triple=riscv32 -mattr=+c,+f -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c,+f < %s \
# RUN:     | llvm-objdump -mattr=+f -riscv-no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: not llvm-mc -triple riscv32 -mattr=+c \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv32 \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv64 -mattr=+c,+f \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s


# CHECK-INST: c.flwsp  fs0, 252(sp)
# CHECK: encoding: [0x7e,0x74]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.flwsp  fs0, 252(sp)
# CHECK-INST: c.fswsp  fa7, 252(sp)
# CHECK: encoding: [0xc6,0xff]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.fswsp  fa7, 252(sp)

# CHECK-INST: c.flw  fa3, 124(a5)
# CHECK: encoding: [0xf4,0x7f]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.flw  fa3, 124(a5)
# CHECK-INST: c.fsw  fa2, 124(a1)
# CHECK: encoding: [0xf0,0xfd]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.fsw  fa2, 124(a1)
