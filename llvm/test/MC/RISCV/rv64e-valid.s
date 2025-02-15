# RUN: llvm-mc %s -triple=riscv64 -M no-aliases -mattr=+e -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+e < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ %s

# This file provides a basic test for RV64E, checking that the expected
# set of registers and instructions are accepted. It only tests instructions
# that are not valid in RV32E.

# CHECK-ASM-AND-OBJ: ld a4, 25(a5)
ld x14, 25(x15)
# CHECK-ASM-AND-OBJ: sd a2, 36(a3)
sd a2, 36(a3)

# CHECK-ASM-AND-OBJ: addiw a4, a5, 37
addiw a4, a5, 37
# CHECK-ASM-AND-OBJ: slliw t1, t1, 31
slliw t1, t1, 31
# CHECK-ASM-AND-OBJ: srliw a0, a4, 0
srliw a0, a4, 0
# CHECK-ASM-AND-OBJ: sraiw a1, sp, 15
sraiw a1, sp, 15
# CHECK-ASM-AND-OBJ: slliw t0, t1, 13
slliw t0, t1, 13

# CHECK-ASM-AND-OBJ: addw ra, zero, zero
addw ra, zero, zero
# CHECK-ASM-AND-OBJ: subw t0, t2, t1
subw t0, t2, t1
# CHECK-ASM-AND-OBJ: sllw a5, a4, a3
sllw a5, a4, a3
# CHECK-ASM-AND-OBJ: srlw a0, s0, t0
srlw a0, s0, t0
# CHECK-ASM-AND-OBJ: sraw t0, a3, zero
sraw t0, a3, zero
