## Arch string without version.

# RUN: llvm-mc %s -triple=riscv32 -filetype=asm | FileCheck %s
# RUN: llvm-mc %s -triple=riscv64 -filetype=asm | FileCheck %s

.attribute arch, "rv32i"
# CHECK: attribute      5, "rv32i2p0"

.attribute arch, "rv32i2"
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

.attribute arch, "rv32ima2p0_fdc"
# CHECK: attribute      5, "rv32i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"

## Experimental extensions require version string to be explicitly specified

.attribute arch, "rv32iv0p10"
# CHECK: attribute      5, "rv32i2p0_v0p10_zvl128b0p10_zvl32b0p10_zvl64b0p10_zvlsseg0p10"

.attribute arch, "rv32izba1p0"
# CHECK: attribute      5, "rv32i2p0_zba1p0"

.attribute arch, "rv32izbb1p0"
# CHECK: attribute      5, "rv32i2p0_zbb1p0"

.attribute arch, "rv32izbc1p0"
# CHECK: attribute      5, "rv32i2p0_zbc1p0"

.attribute arch, "rv32izbe0p93"
# CHECK: attribute      5, "rv32i2p0_zbe0p93"

.attribute arch, "rv32izbf0p93"
# CHECK: attribute      5, "rv32i2p0_zbf0p93"

.attribute arch, "rv32izbm0p93"
# CHECK: attribute      5, "rv32i2p0_zbm0p93"

.attribute arch, "rv32izbp0p93"
# CHECK: attribute      5, "rv32i2p0_zbp0p93"

.attribute arch, "rv32izbr0p93"
# CHECK: attribute      5, "rv32i2p0_zbr0p93"

.attribute arch, "rv32izbs1p0"
# CHECK: attribute      5, "rv32i2p0_zbs1p0"

.attribute arch, "rv32izbt0p93"
# CHECK: attribute      5, "rv32i2p0_zbt0p93"

.attribute arch, "rv32ifzfhmin1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_zfhmin1p0"

.attribute arch, "rv32ifzfh1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_zfh1p0_zfhmin1p0"

.attribute arch, "rv32iv0p10_zvlsseg0p10"
# CHECK: attribute      5, "rv32i2p0_v0p10_zvl128b0p10_zvl32b0p10_zvl64b0p10_zvlsseg0p10"

.attribute arch, "rv32iv0p10zvl32b0p10"
# CHECK: attribute      5, "rv32i2p0_v0p10_zvl128b0p10_zvl32b0p10_zvl64b0p10_zvlsseg0p10"

.attribute arch, "rv32iv0p10zvl64b0p10"
# CHECK: attribute      5, "rv32i2p0_v0p10_zvl128b0p10_zvl32b0p10_zvl64b0p10_zvlsseg0p10"

.attribute arch, "rv32iv0p10zvl128b0p10"
# CHECK: attribute      5, "rv32i2p0_v0p10_zvl128b0p10_zvl32b0p10_zvl64b0p10_zvlsseg0p10"

.attribute arch, "rv32iv0p10zvl256b0p10"
# CHECK: attribute      5, "rv32i2p0_v0p10_zvl128b0p10_zvl256b0p10_zvl32b0p10_zvl64b0p10_zvlsseg0p10"

.attribute arch, "rv32iv0p10zvl512b0p10"
# CHECK: attribute      5, "rv32i2p0_v0p10_zvl128b0p10_zvl256b0p10_zvl32b0p10_zvl512b0p10_zvl64b0p10_zvlsseg0p10"

.attribute arch, "rv32iv0p10zvl1024b0p10"
# CHECK: attribute      5, "rv32i2p0_v0p10_zvl1024b0p10_zvl128b0p10_zvl256b0p10_zvl32b0p10_zvl512b0p10_zvl64b0p10_zvlsseg0p10"

.attribute arch, "rv32iv0p10zvl2048b0p10"
# CHECK: attribute      5, "rv32i2p0_v0p10_zvl1024b0p10_zvl128b0p10_zvl2048b0p10_zvl256b0p10_zvl32b0p10_zvl512b0p10_zvl64b0p10_zvlsseg0p10"

.attribute arch, "rv32iv0p10zvl4096b0p10"
# CHECK: attribute      5, "rv32i2p0_v0p10_zvl1024b0p10_zvl128b0p10_zvl2048b0p10_zvl256b0p10_zvl32b0p10_zvl4096b0p10_zvl512b0p10_zvl64b0p10_zvlsseg0p10"

.attribute arch, "rv32iv0p10zvl8192b0p10"
# CHECK: attribute      5, "rv32i2p0_v0p10_zvl1024b0p10_zvl128b0p10_zvl2048b0p10_zvl256b0p10_zvl32b0p10_zvl4096b0p10_zvl512b0p10_zvl64b0p10_zvl8192b0p10_zvlsseg0p10"

.attribute arch, "rv32iv0p10zvl16384b0p10"
# CHECK: attribute      5, "rv32i2p0_v0p10_zvl1024b0p10_zvl128b0p10_zvl16384b0p10_zvl2048b0p10_zvl256b0p10_zvl32b0p10_zvl4096b0p10_zvl512b0p10_zvl64b0p10_zvl8192b0p10_zvlsseg0p10"

.attribute arch, "rv32iv0p10zvl32768b0p10"
# CHECK: attribute      5, "rv32i2p0_v0p10_zvl1024b0p10_zvl128b0p10_zvl16384b0p10_zvl2048b0p10_zvl256b0p10_zvl32768b0p10_zvl32b0p10_zvl4096b0p10_zvl512b0p10_zvl64b0p10_zvl8192b0p10_zvlsseg0p10"

.attribute arch, "rv32iv0p10zvl65536b0p10"
# CHECK: attribute      5, "rv32i2p0_v0p10_zvl1024b0p10_zvl128b0p10_zvl16384b0p10_zvl2048b0p10_zvl256b0p10_zvl32768b0p10_zvl32b0p10_zvl4096b0p10_zvl512b0p10_zvl64b0p10_zvl65536b0p10_zvl8192b0p10_zvlsseg0p10"
