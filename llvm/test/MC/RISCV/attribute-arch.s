## Arch string without version.

# RUN: llvm-mc %s -triple=riscv32 -filetype=asm | FileCheck %s
# RUN: llvm-mc %s -triple=riscv64 -filetype=asm | FileCheck %s

.attribute arch, "rv32i"
# CHECK: attribute      5, "rv32i2p0"

.attribute arch, "rv32i2"
# CHECK: attribute      5, "rv32i2p0"

.attribute arch, "rv32i2p"
# CHECK: attribute      5, "rv32i2p0"

.attribute arch, "rv32i2p0"
# CHECK: attribute      5, "rv32i2p0"

.attribute arch, "rv32i2_m2"
# CHECK: attribute      5, "rv32i2p0_m2p0"

.attribute arch, "rv32i2_ma"
# CHECK: attribute      5, "rv32i2p0_m2p0_a2p0"

.attribute arch, "rv32g"
# CHECK: attribute      5, "rv32i2p0_m2p0_a2p0_f2p0_d2p0"

.attribute arch, "rv32imafdc"
# CHECK: attribute      5, "rv32i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"

.attribute arch, "rv32i2p0_mafdc"
# CHECK: attribute      5, "rv32i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"

.attribute arch, "rv32ima2p0_fdc"
# CHECK: attribute      5, "rv32i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"

.attribute arch, "rv32ima2p_fdc"
# CHECK: attribute      5, "rv32i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"

.attribute arch, "rv32iv"
# CHECK: attribute      5, "rv32i2p0_v0p10"

.attribute arch, "rv32izba"
# CHECK: attribute      5, "rv32i2p0_zba0p93"

.attribute arch, "rv32izbb"
# CHECK: attribute      5, "rv32i2p0_zbb0p93"

.attribute arch, "rv32izbc"
# CHECK: attribute      5, "rv32i2p0_zbc0p93"

.attribute arch, "rv32izbe"
# CHECK: attribute      5, "rv32i2p0_zbe0p93"

.attribute arch, "rv32izbf"
# CHECK: attribute      5, "rv32i2p0_zbf0p93"

.attribute arch, "rv32izbm"
# CHECK: attribute      5, "rv32i2p0_zbm0p93"

.attribute arch, "rv32izbp"
# CHECK: attribute      5, "rv32i2p0_zbp0p93"

.attribute arch, "rv32izbr"
# CHECK: attribute      5, "rv32i2p0_zbr0p93"

.attribute arch, "rv32izbs"
# CHECK: attribute      5, "rv32i2p0_zbs0p93"

.attribute arch, "rv32izbt"
# CHECK: attribute      5, "rv32i2p0_zbt0p93"

.attribute arch, "rv32ifzfh"
# CHECK: attribute      5, "rv32i2p0_f2p0_zfh0p1"

.attribute arch, "rv32ivzvamo_zvlsseg"
# CHECK: attribute      5, "rv32i2p0_v0p10_zvamo0p10_zvlsseg0p10"

.attribute arch, "rv32iv_zvamo0p10_zvlsseg"
# CHECK: attribute      5, "rv32i2p0_v0p10_zvamo0p10_zvlsseg0p10"
