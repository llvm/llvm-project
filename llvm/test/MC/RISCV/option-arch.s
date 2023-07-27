# RUN: llvm-mc -triple riscv32 -show-encoding < %s \
# RUN:   | FileCheck -check-prefixes=CHECK %s
# RUN: llvm-mc -triple riscv32 -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+c,+m,+a,+f,+zba -d -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-INST %s

# Test '.option arch, +' and '.option arch, -' directive
# The following test cases were copied from MC/RISCV/option-rvc.s

# CHECK-INST: addi a0, a1, 0
# CHECK: # encoding:  [0x13,0x85,0x05,0x00]
addi a0, a1, 0

# CHECK-INST: addi s0, sp, 1020
# CHECK: # encoding:  [0x13,0x04,0xc1,0x3f]
addi s0, sp, 1020

# CHECK: .option arch, +c
.option arch, +c
# CHECK-INST: c.mv a0, a1
# CHECK: # encoding:  [0x2e,0x85]
addi a0, a1, 0

# CHECK-INST: c.addi4spn s0, sp, 1020
# CHECK: # encoding:  [0xe0,0x1f]
addi s0, sp, 1020

# CHECK: .option arch, -c
.option arch, -c
# CHECK-INST: addi a0, a1, 0
# CHECK: # encoding:  [0x13,0x85,0x05,0x00]
addi a0, a1, 0

# CHECK-INST: addi s0, sp, 1020
# CHECK: # encoding:  [0x13,0x04,0xc1,0x3f]
addi s0, sp, 1020

# CHECK: .option arch, +c
.option arch, +c
# CHECK-INST: c.mv a0, a1
# CHECK: # encoding:  [0x2e,0x85]
addi a0, a1, 0

# CHECK-INST: c.addi4spn s0, sp, 1020
# CHECK: # encoding:  [0xe0,0x1f]
addi s0, sp, 1020

# CHECK: .option arch, -c
.option arch, -c
# CHECK-INST: addi a0, a1, 0
# CHECK: # encoding:  [0x13,0x85,0x05,0x00]
addi a0, a1, 0

# CHECK-INST: addi s0, sp, 1020
# CHECK: # encoding:  [0x13,0x04,0xc1,0x3f]
addi s0, sp, 1020

# CHECK: .option arch, +d, -d
.option arch, +d, -d
# CHECK-INST: flw ft0, 12(a0)
# CHECK: # encoding:  [0x07,0x20,0xc5,0x00]
flw f0, 12(a0)

# Test comma-separated list
# CHECK: arch, +m, +a
.option arch, +m, +a
# CHECK-INST: mul a4, ra, s0
# CHECK: # encoding:  [0x33,0x87,0x80,0x02]
mul a4, ra, s0
# CHECK-INST: lr.w t0, (t1)
# CHECK: # encoding:  [0xaf,0x22,0x03,0x10]
lr.w t0, (t1)

# Test multi-letter extension
# CHECK: .option arch, +zba
.option arch, +zba
# CHECK-INST: sh1add t0, t1, t2
# CHECK: encoding: [0xb3,0x22,0x73,0x20]
sh1add t0, t1, t2

# Test '.option arch, <arch-string>' directive
# CHECK: .option arch, rv32i2p1_m2p0_a2p1_c2p0
.option arch, rv32i2p1_m2p0_a2p1_c2p0

# CHECK-INST: mul a4, ra, s0
# CHECK: # encoding:  [0x33,0x87,0x80,0x02]
mul a4, ra, s0
# CHECK-INST: lr.w t0, (t1)
# CHECK: # encoding:  [0xaf,0x22,0x03,0x10]
lr.w t0, (t1)

# Test arch string without version number
# CHECK: .option arch, rv32i2p1_m2p0_a2p1_c2p0
.option arch, rv32imac
# CHECK-INST: mul a4, ra, s0
# CHECK: # encoding:  [0x33,0x87,0x80,0x02]
mul a4, ra, s0
# CHECK-INST: lr.w t0, (t1)
# CHECK: # encoding:  [0xaf,0x22,0x03,0x10]
lr.w t0, (t1)

# Test +c, -c and vice-versa
.option arch, +c, -c
# CHECK-INST: addi a0, a1, 0
# CHECK: # encoding:  [0x13,0x85,0x05,0x00]
addi a0, a1, 0

.option arch, -c, +c
# CHECK-INST: c.mv a0, a1
# CHECK: # encoding:  [0x2e,0x85]
addi a0, a1, 0

.option arch, rv32ic
# CHECK-INST: c.mv a0, a1
# CHECK: # encoding:  [0x2e,0x85]
addi a0, a1, 0

# Test extension name that has digits.
.option arch, +zve32x
# CHECK: .option arch, +zve32x

.option arch, rv32i
.option arch, +zce, +f
# CHECK-INST: flw fa0, 0(a0)
# CHECK: # encoding: [0x08,0x61]
c.flw fa0, 0(a0)
