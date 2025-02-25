# RUN: llvm-mc %s -triple=riscv32 -mattr=+f -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc %s -triple riscv64 -mattr=+f -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+f < %s \
# RUN:     | llvm-objdump --mattr=+f -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+f < %s \
# RUN:     | llvm-objdump --mattr=+f -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s

target:

# CHECK-ASM: .insn r 51, 0, 0, a0, a1, a2
# CHECK-ASM: encoding: [0x33,0x85,0xc5,0x00]
# CHECK-OBJ: add a0, a1, a2
.insn r  0x33,  0,  0, a0, a1, a2
# CHECK-ASM: .insn r 51, 0, 0, a0, a1, a2
# CHECK-ASM: encoding: [0x33,0x85,0xc5,0x00]
# CHECK-OBJ: add a0, a1, a2
.insn r  OP,  0,  0, a0, a1, a2

# CHECK-ASM: .insn i 19, 0, a0, a1, 13
# CHECK-ASM: encoding: [0x13,0x85,0xd5,0x00]
# CHECK-OBJ: addi a0, a1, 0xd
.insn i  0x13,  0, a0, a1, 13
# CHECK-ASM: .insn i 19, 0, a0, a1, 13
# CHECK-ASM: encoding: [0x13,0x85,0xd5,0x00]
# CHECK-OBJ: addi a0, a1, 0xd
.insn i  OP_IMM,  0, a0, a1, 13

# CHECK-ASM: .insn i 103, 0, a0, 10(a1)
# CHECK-ASM: encoding: [0x67,0x85,0xa5,0x00]
# CHECK-OBJ: jalr a0, 0xa(a1)
.insn i  0x67,  0, a0, 10(a1)
# CHECK-ASM: .insn i 103, 0, a0, 10(a1)
# CHECK-ASM: encoding: [0x67,0x85,0xa5,0x00]
# CHECK-OBJ: jalr a0, 0xa(a1)
.insn i  JALR,  0, a0, 10(a1)

# CHECK-ASM: .insn i 103, 0, a0, 0(a1)
# CHECK-ASM: encoding: [0x67,0x85,0x05,0x00]
# CHECK-OBJ: jalr a0, 0x0(a1)
.insn i  0x67,  0, a0, (a1)
# CHECK-ASM: .insn i 103, 0, a0, 0(a1)
# CHECK-ASM: encoding: [0x67,0x85,0x05,0x00]
# CHECK-OBJ: jalr a0, 0x0(a1)
.insn i  JALR,  0, a0, (a1)

# CHECK-ASM: .insn i 3, 0, a0, 4(a1)
# CHECK-ASM: encoding: [0x03,0x85,0x45,0x00]
# CHECK-OBJ: lb a0, 0x4(a1)
.insn i   0x3,  0, a0, 4(a1)
# CHECK-ASM: .insn i 3, 0, a0, 4(a1)
# CHECK-ASM: encoding: [0x03,0x85,0x45,0x00]
# CHECK-OBJ: lb a0, 0x4(a1)
.insn i   LOAD,  0, a0, 4(a1)

# CHECK-ASM: .insn i 3, 0, a0, 0(a1)
# CHECK-ASM: encoding: [0x03,0x85,0x05,0x00]
# CHECK-OBJ: lb a0, 0x0(a1)
.insn i   0x3,  0, a0, (a1)
# CHECK-ASM: .insn i 3, 0, a0, 0(a1)
# CHECK-ASM: encoding: [0x03,0x85,0x05,0x00]
# CHECK-OBJ: lb a0, 0x0(a1)
.insn i   LOAD,  0, a0, (a1)

# CHECK-ASM: .insn b 99, 0, a0, a1, target
# CHECK-ASM: [0x63'A',A,0xb5'A',A]
# CHECK-OBJ: beq a0, a1, 0x0 <target>
.insn sb 0x63,  0, a0, a1, target
# CHECK-ASM: .insn b 99, 0, a0, a1, target
# CHECK-ASM: [0x63'A',A,0xb5'A',A]
# CHECK-OBJ: beq a0, a1, 0x0 <target>
.insn sb BRANCH,  0, a0, a1, target

# CHECK-ASM: .insn b 99, 0, a0, a1, target
# CHECK-ASM: [0x63'A',A,0xb5'A',A]
# CHECK-OBJ: beq a0, a1, 0x0 <target>
.insn b  0x63,  0, a0, a1, target
# CHECK-ASM: .insn b 99, 0, a0, a1, target
# CHECK-ASM: [0x63'A',A,0xb5'A',A]
# CHECK-OBJ: beq a0, a1, 0x0 <target>
.insn b  BRANCH,  0, a0, a1, target

# CHECK-ASM: .insn s 35, 0, a0, 4(a1)
# CHECK-ASM: encoding: [0x23,0x82,0xa5,0x00]
# CHECK-OBJ: sb a0, 0x4(a1)
.insn s  0x23,  0, a0, 4(a1)
# CHECK-ASM: .insn s 35, 0, a0, 4(a1)
# CHECK-ASM: encoding: [0x23,0x82,0xa5,0x00]
# CHECK-OBJ: sb a0, 0x4(a1)
.insn s  STORE,  0, a0, 4(a1)

# CHECK-ASM: .insn s 35, 0, a0, 0(a1)
# CHECK-ASM: encoding: [0x23,0x80,0xa5,0x00]
# CHECK-OBJ: sb a0, 0x0(a1)
.insn s  0x23,  0, a0, (a1)
# CHECK-ASM: .insn s 35, 0, a0, 0(a1)
# CHECK-ASM: encoding: [0x23,0x80,0xa5,0x00]
# CHECK-OBJ: sb a0, 0x0(a1)
.insn s  STORE,  0, a0, (a1)

# CHECK-ASM: .insn u 55, a0, 4095
# CHECK-ASM: encoding: [0x37,0xf5,0xff,0x00]
# CHECK-OBJ: lui a0, 0xfff
.insn u  0x37, a0, 0xfff
# CHECK-ASM: .insn u 55, a0, 4095
# CHECK-ASM: encoding: [0x37,0xf5,0xff,0x00]
# CHECK-OBJ: lui a0, 0xfff
.insn u  LUI, a0, 0xfff

# CHECK-ASM: .insn j 111, a0, target
# CHECK-ASM: encoding: [0x6f,0bAAAA0101,A,A]
# CHECK-OBJ: jal a0, 0x0 <target>
.insn uj 0x6f, a0, target
# CHECK-ASM: .insn j 111, a0, target
# CHECK-ASM: encoding: [0x6f,0bAAAA0101,A,A]
# CHECK-OBJ: jal a0, 0x0 <target>
.insn uj JAL, a0, target

# CHECK-ASM: .insn j 111, a0, target
# CHECK-ASM: encoding: [0x6f,0bAAAA0101,A,A]
# CHECK-OBJ: jal a0, 0x0 <target>
.insn j  0x6f, a0, target
# CHECK-ASM: .insn j 111, a0, target
# CHECK-ASM: encoding: [0x6f,0bAAAA0101,A,A]
# CHECK-OBJ: jal a0, 0x0 <target>
.insn j  JAL, a0, target

# CHECK-ASM: .insn r4 67, 0, 0, fa0, fa1, fa2, fa3
# CHECK-ASM: encoding: [0x43,0x85,0xc5,0x68]
# CHECK-OBJ: fmadd.s fa0, fa1, fa2, fa3, rne
.insn r  0x43,  0,  0, fa0, fa1, fa2, fa3
# CHECK-ASM: .insn r4 67, 0, 0, fa0, fa1, fa2, fa3
# CHECK-ASM: encoding: [0x43,0x85,0xc5,0x68]
# CHECK-OBJ: fmadd.s fa0, fa1, fa2, fa3, rne
.insn r  MADD,  0,  0, fa0, fa1, fa2, fa3

# CHECK-ASM: .insn r4 67, 0, 0, fa0, fa1, fa2, fa3
# CHECK-ASM: encoding: [0x43,0x85,0xc5,0x68]
# CHECK-OBJ: fmadd.s fa0, fa1, fa2, fa3, rne
.insn r4 0x43,  0,  0, fa0, fa1, fa2, fa3
# CHECK-ASM: .insn r4 67, 0, 0, fa0, fa1, fa2, fa3
# CHECK-ASM: encoding: [0x43,0x85,0xc5,0x68]
# CHECK-OBJ: fmadd.s fa0, fa1, fa2, fa3, rne
.insn r4 MADD,  0,  0, fa0, fa1, fa2, fa3

# CHECK-ASM: .insn i 3, 5, t1, -2048(t2)
# CHECK-ASM: encoding: [0x03,0xd3,0x03,0x80]
# CHECK-OBJ: lhu t1, -0x800(t2)
.insn i 0x3, 0x5, x6, %lo(2048)(x7)
# CHECK-ASM: .insn i 3, 5, t1, -2048(t2)
# CHECK-ASM: encoding: [0x03,0xd3,0x03,0x80]
# CHECK-OBJ: lhu t1, -0x800(t2)
.insn i LOAD, 0x5, x6, %lo(2048)(x7)

# CHECK-ASM: .insn 0x4, 19
# CHECK-ASM: encoding: [0x13,0x00,0x00,0x00]
# CHECK-OBJ: addi zero, zero, 0x0
.insn 0x13

# CHECK-ASM: .insn 0x4, 19
# CHECK-ASM: encoding: [0x13,0x00,0x00,0x00]
# CHECK-OBJ: addi zero, zero, 0x0
.insn 0x4, 0x13

# CHECK-ASM: .insn 0x6, 31
# CHECK-ASM: encoding: [0x1f,0x00,0x00,0x00,0x00,0x00]
# CHECK-OBJ: <unknown>
.insn 6, 0x1f

# CHECK-ASM: .insn 0x8, 63
# CHECK-ASM: encoding: [0x3f,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
# CHECK-OBJ: <unknown>
.insn 8, 0x3f

# CHECK-ASM: .insn 0x6, 281474976710623
# CHECK-ASM: encoding: [0xdf,0xff,0xff,0xff,0xff,0xff]
# CHECK-OBJ: <unknown>
.insn 0x6, 0xffffffffffdf

# CHECK-ASM: .insn 0x8, -65
# CHECK-ASM: encoding: [0xbf,0xff,0xff,0xff,0xff,0xff,0xff,0xff]
# CHECK-OBJ: <unknown>
.insn 0x8, 0xffffffffffffffbf

# CHECK-ASM: .insn 0x4, 3971
# CHECK-ASM: encoding: [0x83,0x0f,0x00,0x00]
# CHECK-OBJ: lb t6, 0x0(zero)
.insn 0x2 + 0x2, 0x3 | (31 << 7)

# CHECK-ASM: .insn 0x8, -576460752303423297
# CHECK-ASM: encoding: [0xbf,0x00,0x00,0x00,0x00,0x00,0x00,0xf8]
# CHECK-OBJ: <unknown>
.insn 0x4 * 0x2, 0xbf | (31 << 59)

odd_lengths:
# CHECK-ASM-LABEL: odd_lengths:
# CHECK-OBJ-LABEL: <odd_lengths>:

## These deliberately disagree with the lengths objdump expects them to have, so
## keep them at the end so that the disassembled instruction stream is not out
## of sync with the encoded instruction stream. We don't check for `<unknown>`
## as we could get any number of those, so instead check for the encoding
## halfwords. These might be split into odd 16-bit chunks, so each chunk is on
## one line.

# CHECK-ASM: .insn 0x4, 65503
# CHECK-ASM: encoding: [0xdf,0xff,0x00,0x00]
# CHECK-OBJ: ffdf
# CHECK-OBJ: 0000
.insn 0xffdf

# CHECK-ASM: .insn 0x4, 65471
# CHECK-ASM: encoding: [0xbf,0xff,0x00,0x00]
# CHECK-OBJ: ffbf
# CHECK-OBJ: 0000
.insn 0xffbf
