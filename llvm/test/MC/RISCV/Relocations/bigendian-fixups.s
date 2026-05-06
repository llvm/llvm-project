# RUN: llvm-mc --triple=riscv32be %s --show-encoding \
# RUN:     | FileCheck --check-prefixes=CHECK-FIXUP,CHECK-ENCODING %s
# RUN: llvm-mc --filetype=obj --triple=riscv32be %s \
# RUN:     | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-mc --filetype=obj --triple=riscv32be %s \
# RUN:     | llvm-readobj -r - | FileCheck --check-prefix=CHECK-REL %s

# RUN: llvm-mc --triple=riscv64be %s --show-encoding \
# RUN:     | FileCheck --check-prefixes=CHECK-FIXUP,CHECK-ENCODING %s
# RUN: llvm-mc --filetype=obj --triple=riscv64be %s \
# RUN:     | llvm-objdump -d - | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-mc --filetype=obj --triple=riscv64be %s \
# RUN:     | llvm-readobj -r - | FileCheck --check-prefix=CHECK-REL %s

## Checks that fixups that can be resolved within the same object file are
## applied correctly on big-endian RISC-V targets.
##
## This test verifies that RISC-V instructions remain little-endian even on
## big-endian systems. This is a fundamental property of RISC-V:
## - Instructions are always little-endian
## - Data can be big-endian or little-endian depending on the system

.LBB0:
addi t0, t0, 1
# CHECK-ENCODING: encoding: [0x93,0x82,0x12,0x00]
# CHECK-INSTR: addi t0, t0, 0x1

lui t1, %hi(val)
# CHECK-ENCODING: encoding: [0x37,0bAAAA0011,A,A]
# CHECK-FIXUP: fixup A - offset: 0, value: %hi(val), kind: fixup_riscv_hi20
# CHECK-INSTR: lui t1, 0x12345

lw a0, %lo(val)(t1)
# CHECK-ENCODING: encoding: [0x03,0x25,0bAAAA0011,A]
# CHECK-FIXUP: fixup A - offset: 0, value: %lo(val), kind: fixup_riscv_lo12_i
# CHECK-INSTR: lw a0, 0x678(t1)

addi a1, t1, %lo(val)
# CHECK-ENCODING: encoding: [0x93,0x05,0bAAAA0011,A]
# CHECK-FIXUP: fixup A - offset: 0, value: %lo(val), kind: fixup_riscv_lo12_i
# CHECK-INSTR: addi a1, t1, 0x678

sw a0, %lo(val)(t1)
# CHECK-ENCODING: encoding: [0x23'A',0x20'A',0xa3'A',A]
# CHECK-FIXUP: fixup A - offset: 0, value: %lo(val), kind: fixup_riscv_lo12_s
# CHECK-INSTR: sw a0, 0x678(t1)

1:
auipc t1, %pcrel_hi(.LBB0)
# CHECK-ENCODING: encoding: [0x17,0bAAAA0011,A,A]
# CHECK-FIXUP: fixup A - offset: 0, value: %pcrel_hi(.LBB0), kind: fixup_riscv_pcrel_hi20
# CHECK-INSTR: auipc t1, 0

addi t1, t1, %pcrel_lo(1b)
# CHECK-ENCODING: encoding: [0x13,0x03,0bAAAA0011,A]
# CHECK-FIXUP: fixup A - offset: 0, value: %pcrel_lo({{.*}}), kind: fixup_riscv_pcrel_lo12_i
# CHECK-INSTR: addi t1, t1, -0x14

sw t1, %pcrel_lo(1b)(t1)
# CHECK-ENCODING: encoding: [0x23'A',0x20'A',0x63'A',A]
# CHECK-FIXUP: fixup A - offset: 0, value: %pcrel_lo({{.*}}), kind: fixup_riscv_pcrel_lo12_s
# CHECK-INSTR: sw t1, -0x14(t1)

jal zero, .LBB0
# CHECK-ENCODING: encoding: [0x6f,0bAAAA0000,A,A]
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB0, kind: fixup_riscv_jal
# CHECK-INSTR: j 0x0 <.text>

jal zero, .LBB2
# CHECK-ENCODING: encoding: [0x6f,0bAAAA0000,A,A]
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB2, kind: fixup_riscv_jal
# CHECK-INSTR: j 0x50d18 <.text+0x50d18>

beq a0, a1, .LBB0
# CHECK-ENCODING: encoding: [0x63'A',A,0xb5'A',A]
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB0, kind: fixup_riscv_branch
# CHECK-INSTR: beq a0, a1, 0x0 <.text>

blt a0, a1, .LBB1
# CHECK-ENCODING: encoding: [0x63'A',0x40'A',0xb5'A',A]
# CHECK-FIXUP: fixup A - offset: 0, value: .LBB1, kind: fixup_riscv_branch
# CHECK-INSTR: blt a0, a1, 0x480 <.text+0x480>

.fill 1104

.LBB1:

.fill 329876
addi zero, zero, 0
.LBB2:

.set val, 0x12345678

# CHECK-REL-NOT: R_RISCV

.data
.align 3
data_label:
  .word val  # On BE: 0x12345678 stored as [0x12, 0x34, 0x56, 0x78]
  .long val  # On BE: 0x12345678 stored as [0x12, 0x34, 0x56, 0x78]
  .quad val  # On BE: 0x0000000012345678 stored as [0x00, 0x00, 0x00, 0x00, 0x12, 0x34, 0x56, 0x78]
