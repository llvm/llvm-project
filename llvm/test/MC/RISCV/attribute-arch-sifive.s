## Arch string without version.

# RUN: llvm-mc %s -triple=riscv32 -filetype=asm | FileCheck %s
# RUN: llvm-mc %s -triple=riscv64 -filetype=asm | FileCheck %s

.attribute arch, "rv32i_xsfmm128t"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl512b1p0_zvl64b1p0_xsfmm128t0p6_xsfmmbase0p6"

.attribute arch, "rv32i_xsfmm16t"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvl32b1p0_zvl64b1p0_xsfmm16t0p6_xsfmmbase0p6"

.attribute arch, "rv32i_xsfmm32a8i"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvl32b1p0_xsfmm32a8i0p6_xsfmmbase0p6"

.attribute arch, "rv32i_xsfmm32a8f"
# CHECK: attribute      5, "rv32i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvl32b1p0_xsfmm32a8f0p6_xsfmmbase0p6"

.attribute arch, "rv32i_xsfmm32a16f"
# CHECK: attribute      5, "rv32i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvl32b1p0_xsfmm32a16f0p6_xsfmmbase0p6"

.attribute arch, "rv32i_xsfmm32a32f"
# CHECK: attribute      5, "rv32i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvl32b1p0_xsfmm32a32f0p6_xsfmmbase0p6"

.attribute arch, "rv32i_xsfmm32t"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0_xsfmm32t0p6_xsfmmbase0p6"

.attribute arch, "rv32i_xsfmm64a64f"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl32b1p0_zvl64b1p0_xsfmm64a64f0p6_xsfmmbase0p6"

.attribute arch, "rv32i_xsfmm64t"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl64b1p0_xsfmm64t0p6_xsfmmbase0p6"

.attribute arch, "rv32i_xsfmmbase"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvl32b1p0_xsfmmbase0p6"
