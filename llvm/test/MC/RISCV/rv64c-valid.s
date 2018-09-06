# RUN: llvm-mc %s -triple=riscv64 -mattr=+c -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c < %s \
# RUN:     | llvm-objdump -mattr=+c -riscv-no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s
#
#
# RUN: not llvm-mc -triple riscv64 \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv32 -mattr=+c \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s

# FIXME: error messages for rv32c are misleading

# TODO: more exhaustive testing of immediate encoding.

# CHECK-ASM-AND-OBJ: c.ldsp ra, 0(sp)
# CHECK-ASM: encoding: [0x82,0x60]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.ldsp ra, 0(sp)
# CHECK-ASM-AND-OBJ: c.sdsp ra, 504(sp)
# CHECK-ASM: encoding: [0x86,0xff]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.sdsp ra, 504(sp)
# CHECK-ASM-AND-OBJ: c.ld a4, 0(a3)
# CHECK-ASM: encoding: [0x98,0x62]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.ld a4, 0(a3)
# CHECK-ASM-AND-OBJ: c.sd a5, 248(a3)
# CHECK-ASM: encoding: [0xfc,0xfe]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.sd a5, 248(a3)

# CHECK-ASM-AND-OBJ: c.subw a3, a4
# CHECK-ASM: encoding: [0x99,0x9e]
c.subw a3, a4
# CHECK-ASM-AND-OBJ: c.addw a0, a2
# CHECK-ASM: encoding: [0x31,0x9d]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.addw a0, a2

# CHECK-ASM-AND-OBJ: c.addiw a3, -32
# CHECK-ASM: encoding: [0x81,0x36]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.addiw a3, -32
# CHECK-ASM-AND-OBJ: c.addiw a3, 31
# CHECK-ASM: encoding: [0xfd,0x26]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.addiw a3, 31

# CHECK-ASM-AND-OBJ: c.slli s0, 1
# CHECK-ASM: encoding: [0x06,0x04]
# CHECK-NO-EXT:  error: instruction use requires an option to be enabled
c.slli s0, 1
# CHECK-ASM-AND-OBJ: c.srli a3, 63
# CHECK-ASM: encoding: [0xfd,0x92]
c.srli a3, 63
# CHECK-ASM-AND-OBJ: c.srai a2, 63
# CHECK-ASM: encoding: [0x7d,0x96]
c.srai a2, 63
