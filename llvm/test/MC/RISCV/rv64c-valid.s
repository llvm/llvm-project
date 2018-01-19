# RUN: llvm-mc -triple=riscv64 -mattr=+c -riscv-no-aliases -show-encoding < %s \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+c < %s \
# RUN:     | llvm-objdump -mattr=+c -riscv-no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: not llvm-mc -triple riscv64 \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv32 -mattr=+c\
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s

# TODO: more exhaustive testing of immediate encoding.

# CHECK-INST: c.ldsp  ra, 0(sp)
# CHECK: encoding: [0x82,0x60]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.ldsp  ra, 0(sp)
# CHECK-INST: c.sdsp  ra, 504(sp)
# CHECK: encoding: [0x86,0xff]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.sdsp  ra, 504(sp)
# CHECK-INST: c.ld    a4, 0(a3)
# CHECK: encoding: [0x98,0x62]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.ld    a4, 0(a3)
# CHECK-INST: c.sd    a5, 248(a3)
# CHECK: encoding: [0xfc,0xfe]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.sd    a5, 248(a3)

# CHECK-INST: c.subw   a3, a4
# CHECK: encoding: [0x99,0x9e]
c.subw   a3, a4
# CHECK-INST: c.addw   a0, a2
# CHECK: encoding: [0x31,0x9d]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.addw   a0, a2

# CHECK-INST: c.addiw  a3, -32
# CHECK: encoding: [0x81,0x36]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.addiw  a3, -32
# CHECK-INST: c.addiw  a3, 31
# CHECK: encoding: [0xfd,0x26]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.addiw  a3, 31

# CHECK-INST: c.slli  s0, 1
# CHECK: encoding: [0x06,0x04]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.slli  s0, 1
# CHECK-INST: c.srli  a3, 63
# CHECK: encoding: [0xfd,0x92]
c.srli  a3, 63
# CHECK-INST: c.srai  a2, 63
# CHECK: encoding: [0x7d,0x96]
c.srai  a2, 63
