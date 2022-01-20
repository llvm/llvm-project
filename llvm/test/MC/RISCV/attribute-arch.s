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

.attribute arch, "rv32iv1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

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

.attribute arch, "rv32iv1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32iv1p0zvl32b1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32iv1p0zvl64b1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32iv1p0zvl128b1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32iv1p0zvl256b1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32iv1p0zvl512b1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl512b1p0_zvl64b1p0"

.attribute arch, "rv32iv1p0zvl1024b1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl512b1p0_zvl64b1p0"

.attribute arch, "rv32iv1p0zvl2048b1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl2048b1p0_zvl256b1p0_zvl32b1p0_zvl512b1p0_zvl64b1p0"

.attribute arch, "rv32iv1p0zvl4096b1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl2048b1p0_zvl256b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0"

.attribute arch, "rv32iv1p0zvl8192b1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl2048b1p0_zvl256b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0_zvl8192b1p0"

.attribute arch, "rv32iv1p0zvl16384b1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl16384b1p0_zvl2048b1p0_zvl256b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0_zvl8192b1p0"

.attribute arch, "rv32iv1p0zvl32768b1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl16384b1p0_zvl2048b1p0_zvl256b1p0_zvl32768b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0_zvl8192b1p0"

.attribute arch, "rv32iv1p0zvl65536b1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl16384b1p0_zvl2048b1p0_zvl256b1p0_zvl32768b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0_zvl65536b1p0_zvl8192b1p0"

.attribute arch, "rv32i_zve32x1p0"
# CHECK: attribute      5, "rv32i2p0_zve32x1p0_zvl32b1p0"

.attribute arch, "rv32if_zve32f1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_zve32f1p0_zve32x1p0_zvl32b1p0"

.attribute arch, "rv32i_zve64x1p0"
# CHECK: attribute      5, "rv32i2p0_zve32x1p0_zve64x1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32if_zve64f1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_zve32f1p0_zve32x1p0_zve64f1p0_zve64x1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32ifd_zve64d1p0"
# CHECK: attribute      5, "rv32i2p0_f2p0_d2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32i_zbkb1p0"
# CHECK: attribute      5, "rv32i2p0_zbkb1p0"
