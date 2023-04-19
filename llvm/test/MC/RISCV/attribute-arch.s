## Arch string without version.

# RUN: llvm-mc %s -triple=riscv32 -filetype=asm | FileCheck %s
# RUN: llvm-mc %s -triple=riscv64 -filetype=asm \
# RUN:     | FileCheck --check-prefixes=CHECK,CHECK-RV64 %s

.attribute arch, "rv32i"
# CHECK: attribute      5, "rv32i2p1"

.attribute arch, "rv32i2p1"
# CHECK: attribute      5, "rv32i2p1"

.attribute arch, "rv32e"
# CHECK: attribute      5, "rv32e2p0"

.attribute arch, "rv64e"
# CHECK-RV64: attribute      5, "rv64e2p0"

.attribute arch, "rv32i2p1_m2"
# CHECK: attribute      5, "rv32i2p1_m2p0"

.attribute arch, "rv32i2p1_ma"
# CHECK: attribute      5, "rv32i2p1_m2p0_a2p1"

.attribute arch, "rv32g"
# CHECK: attribute      5, "rv32i2p1_m2p0_a2p1_f2p2_d2p2_zicsr2p0_zifencei2p0"

.attribute arch, "rv32imafdc"
# CHECK: attribute      5, "rv32i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0"

.attribute arch, "rv32i2p1_mafdc"
# CHECK: attribute      5, "rv32i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0"

.attribute arch, "rv32ima2p1_fdc"
# CHECK: attribute      5, "rv32i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0"

.attribute arch, "rv32ima2p1_fdc"
# CHECK: attribute      5, "rv32i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0"

.attribute arch, "rv32iv"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl32b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl64b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl128b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl256b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl512b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl512b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl1024b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl512b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl2048b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl2048b1p0_zvl256b1p0_zvl32b1p0_zvl512b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl4096b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl2048b1p0_zvl256b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0"

.attribute arch, "rv32ivzvl8192b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl2048b1p0_zvl256b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0_zvl8192b1p0"

.attribute arch, "rv32ivzvl16384b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl16384b1p0_zvl2048b1p0_zvl256b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0_zvl8192b1p0"

.attribute arch, "rv32ivzvl32768b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl16384b1p0_zvl2048b1p0_zvl256b1p0_zvl32768b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0_zvl8192b1p0"

.attribute arch, "rv32ivzvl65536b"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl1024b1p0_zvl128b1p0_zvl16384b1p0_zvl2048b1p0_zvl256b1p0_zvl32768b1p0_zvl32b1p0_zvl4096b1p0_zvl512b1p0_zvl64b1p0_zvl65536b1p0_zvl8192b1p0"

.attribute arch, "rv32izve32x"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvl32b1p0"

.attribute arch, "rv32ifzve32f"
# CHECK: attribute      5, "rv32i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zvl32b1p0"

.attribute arch, "rv32izve64x"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32ifzve64f"
# CHECK: attribute      5, "rv32i2p1_f2p2_zicsr2p0_zve32f1p0_zve32x1p0_zve64f1p0_zve64x1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32ifdzve64d"
# CHECK: attribute      5, "rv32i2p1_f2p2_d2p2_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32izicbom"
# CHECK: attribute      5, "rv32i2p1_zicbom1p0"

.attribute arch, "rv32izicboz"
# CHECK: attribute      5, "rv32i2p1_zicboz1p0"

.attribute arch, "rv32izicbop"
# CHECK: attribute      5, "rv32i2p1_zicbop1p0"

## Experimental extensions require version string to be explicitly specified

.attribute arch, "rv32izba1p0"
# CHECK: attribute      5, "rv32i2p1_zba1p0"

.attribute arch, "rv32izbb1p0"
# CHECK: attribute      5, "rv32i2p1_zbb1p0"

.attribute arch, "rv32izbc1p0"
# CHECK: attribute      5, "rv32i2p1_zbc1p0"

.attribute arch, "rv32i_zve64x_zvkb0p3"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvkb0p3_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32i_zve32x_zvkg0p3"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvkg0p3_zvl32b1p0"

.attribute arch, "rv32i_zve64x_zvkn0p3"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvkb0p3_zvkn0p3_zvkned0p3_zvknha0p3_zvknhb0p3_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32i_zve32x_zvknha0p3"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvknha0p3_zvl32b1p0"

.attribute arch, "rv32i_zve64x_zvknhb0p3"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zve64x1p0_zvknha0p3_zvknhb0p3_zvl32b1p0_zvl64b1p0"

.attribute arch, "rv32i_zve32x_zvkned0p3"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvkned0p3_zvl32b1p0"

.attribute arch, "rv32i_zve32x_zvks0p3"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvkb0p3_zvks0p3_zvksed0p3_zvksh0p3_zvl32b1p0"

.attribute arch, "rv32i_zve32x_zvksed0p3"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvksed0p3_zvl32b1p0"

.attribute arch, "rv32i_zve32x_zvksh0p3"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zve32x1p0_zvksh0p3_zvl32b1p0"

.attribute arch, "rv32izbs1p0"
# CHECK: attribute      5, "rv32i2p1_zbs1p0"

.attribute arch, "rv32ifzfhmin1p0"
# CHECK: attribute      5, "rv32i2p1_f2p2_zicsr2p0_zfhmin1p0"

.attribute arch, "rv32ifzfh1p0"
# CHECK: attribute      5, "rv32i2p1_f2p2_zicsr2p0_zfh1p0"

.attribute arch, "rv32izfinx"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zfinx1p0"

.attribute arch, "rv32izfinx_zdinx"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zfinx1p0_zdinx1p0"

.attribute arch, "rv32izfinx_zhinxmin"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zfinx1p0_zhinxmin1p0"

.attribute arch, "rv32izfinx_zhinx1p0"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0_zfinx1p0_zhinx1p0"

.attribute arch, "rv32i_zbkb1p0"
# CHECK: attribute      5, "rv32i2p1_zbkb1p0"

.attribute arch, "rv32i_zbkc1p0"
# CHECK: attribute      5, "rv32i2p1_zbkc1p0"

.attribute arch, "rv32i_zbkx1p0"
# CHECK: attribute      5, "rv32i2p1_zbkx1p0"

.attribute arch, "rv32i_zknd1p0"
# CHECK: attribute      5, "rv32i2p1_zknd1p0"

.attribute arch, "rv32i_zkne1p0"
# CHECK: attribute      5, "rv32i2p1_zkne1p0"

.attribute arch, "rv32i_zknh1p0"
# CHECK: attribute      5, "rv32i2p1_zknh1p0"

.attribute arch, "rv32i_zksed1p0"
# CHECK: attribute      5, "rv32i2p1_zksed1p0"

.attribute arch, "rv32i_zksh1p0"
# CHECK: attribute      5, "rv32i2p1_zksh1p0"

.attribute arch, "rv32i_zkr1p0"
# CHECK: attribute      5, "rv32i2p1_zkr1p0"

.attribute arch, "rv32i_zkn1p0"
# CHECK: attribute      5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0"

.attribute arch, "rv32i_zks1p0"
# CHECK: attribute      5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zks1p0_zksed1p0_zksh1p0"

.attribute arch, "rv32i_zkt1p0"
# CHECK: attribute      5, "rv32i2p1_zkt1p0"

.attribute arch, "rv32i_zk1p0"
# CHECK: attribute      5, "rv32i2p1_zbkb1p0_zbkc1p0_zbkx1p0_zk1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0_zkr1p0_zkt1p0"

.attribute arch, "rv32izihintntl0p2"
# CHECK: attribute      5, "rv32i2p1_zihintntl0p2"

.attribute arch, "rv32iczihintntl0p2"
# CHECK: attribute      5, "rv32i2p1_c2p0_zihintntl0p2"

.attribute arch, "rv32if_zkt1p0_zve32f1p0_zve32x1p0_zvl32b1p0"
# CHECK: attribute      5, "rv32i2p1_f2p2_zicsr2p0_zkt1p0_zve32f1p0_zve32x1p0_zvl32b1p0"

.attribute arch, "rv32izca1p0"
# CHECK: attribute      5, "rv32i2p1_zca1p0"

.attribute arch, "rv32izcd1p0"
# CHECK: attribute      5, "rv32i2p1_zcd1p0"

.attribute arch, "rv32izcf1p0"
# CHECK: attribute      5, "rv32i2p1_zcf1p0"

.attribute arch, "rv32izcb1p0"
# CHECK: attribute      5, "rv32i2p1_zca1p0_zcb1p0"

.attribute arch, "rv64i_xsfvcp"
# CHECK: attribute      5, "rv64i2p1_zicsr2p0_zve32x1p0_zvl32b1p0_xsfvcp1p0"

.attribute arch, "rv32izawrs1p0"
# CHECK: attribute      5, "rv32i2p1_zawrs1p0"

.attribute arch, "rv32iztso0p1"
# CHECK: attribute      5, "rv32i2p1_ztso0p1"

.attribute arch, "rv32izicsr2p0"
# CHECK: attribute      5, "rv32i2p1_zicsr2p0"

.attribute arch, "rv32izifencei2p0"
# CHECK: attribute      5, "rv32i2p1_zifencei2p0"

.attribute arch, "rv32izfa0p2"
# CHECK: attribute      5, "rv32i2p1_f2p2_zicsr2p0_zfa0p2"

.attribute arch, "rv32izicond1p0"
# CHECK: attribute      5, "rv32i2p1_zicond1p0"
