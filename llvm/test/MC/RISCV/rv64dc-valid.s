# RUN: llvm-mc %s -triple=riscv64 -mattr=+c,+d -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+d < %s \
# RUN:     | llvm-objdump -mattr=+d -riscv-no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: not llvm-mc -triple riscv64 -mattr=+c\
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN: | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN:     not llvm-mc -triple riscv64 \
# RUN: -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s

# CHECK-INST: c.fldsp  fs0, 504(sp)
# CHECK: encoding: [0x7e,0x34]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.fldsp  fs0, 504(sp)
# CHECK-INST: c.fsdsp  fa7, 504(sp)
# CHECK: encoding: [0xc6,0xbf]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.fsdsp  fa7, 504(sp)

# CHECK-INST: c.fld  fa3, 248(a5)
# CHECK: encoding: [0xf4,0x3f]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.fld  fa3, 248(a5)
# CHECK-INST: c.fsd  fa2, 248(a1)
# CHECK: encoding: [0xf0,0xbd]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.fsd  fa2, 248(a1)
