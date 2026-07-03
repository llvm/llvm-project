# RUN: llvm-mc %s -triple=riscv32 -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 < %s \
# RUN:     | llvm-objdump -M no-aliases --no-print-imm-hex -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: addi a0, a1, -1
# CHECK-ASM: encoding: [0x13,0x85,0xf5,0xff]
addi a0, a1, 4294967295 # 0xffffffff
# CHECK-ASM-AND-OBJ: ori a2, a3, -2048
# CHECK-ASM: encoding: [0x13,0xe6,0x06,0x80]
ori a2, a3, 0xfffff800
# CHECK-ASM-AND-OBJ: lw a1, -1621(a2)
# CHECK-ASM: encoding: [0x83,0x25,0xb6,0x9a]
lw a1, 0xfffff9ab(a2)
# CHECK-ASM-AND-OBJ: sw a1, -8(a2)
# CHECK-ASM: encoding: [0x23,0x2c,0xb6,0xfe]
sw a1, 0xfffffff8(a2)

## Branch and Jump immediates are relative but printed as their absolute address
## when disassembling.

# CHECK-ASM: beq t0, t1, -4096
# CHECK-ASM: encoding: [0x63,0x80,0x62,0x80]
# CHECK-OBJ: beq t0, t1, 0xfffff010
beq t0, t1, 0xfffff000

# CHECK-ASM: bne t1, t2, -4082
# CHECK-ASM: encoding: [0x63,0x17,0x73,0x80]
# CHECK-OBJ: bne t1, t2, 0xfffff022
bne t1, t2, 0xfffff00e

# CHECK-ASM: beq t2, zero, -3550
# CHECK-ASM: encoding: [0x63,0x81,0x03,0xa2]
# CHECK-OBJ: beq t2, zero, 0xfffff23a
beqz t2, 0xfffff222

# CHECK-ASM: .insn b 99, 0, a0, a1, -3004
# CHECK-ASM: encoding: [0x63,0x02,0xb5,0xc4]
# CHECK-OBJ: beq a0, a1, 0xfffff460
.insn b  BRANCH,  0, a0, a1, 0xfffff444

# CHECK-ASM: jal ra, -2458
# CHECK-ASM: encoding: [0xef,0xf0,0x6f,0xe6]
# CHECK-OBJ: jal ra, 0xfffff686
jal ra, 0xfffff666

# CHECK-ASM: jal ra, -1912
# CHECK-ASM: encoding: [0xef,0xf0,0x9f,0x88]
# CHECK-OBJ: jal ra, 0xfffff8ac
jal 0xfffff888

# CHECK-ASM: jal zero, -1366
# CHECK-ASM: encoding: [0x6f,0xf0,0xbf,0xaa]
# CHECK-OBJ: jal zero, 0xfffffad2
j 0xfffffaaa

# CHECK-ASM: .insn j 111, a0, -820
# CHECK-ASM: encoding: [0x6f,0x65,0xe6,0xff]
# CHECK-OBJ: jal a0, 0xfff6682a
.insn j JAL, a0, 0xfffffccc
