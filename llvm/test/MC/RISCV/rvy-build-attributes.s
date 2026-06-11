# RUN: llvm-mc %s -triple=riscv64 -filetype=asm -riscv-add-build-attributes \
# RUN:   -mattr=+experimental-y,+c,+d | FileCheck %s --check-prefixes=RV64Y-CD
# RUN: llvm-mc %s -triple=riscv32 -filetype=asm -riscv-add-build-attributes \
# RUN:   -mattr=+experimental-y,+c,+f | FileCheck %s --check-prefixes=RV32Y-CF
# RUN: llvm-mc %s -triple=riscv32 -filetype=asm -riscv-add-build-attributes \
# RUN:   -mattr=+experimental-y,+zca,+f | FileCheck %s --check-prefixes=RV32Y-ZCA-F
# RUN: llvm-mc %s -triple=riscv64 -filetype=asm -riscv-add-build-attributes \
# RUN:   -mattr=+experimental-y,+zca,+d | FileCheck %s --check-prefixes=RV64Y-ZCA-D
# RUN: llvm-mc %s -triple=riscv64 -filetype=asm -riscv-add-build-attributes \
# RUN:   -mattr=+experimental-y,+zca,+zcb | FileCheck %s --check-prefixes=RV64Y-ZCA-ZCB
# RUN: llvm-mc %s -triple=riscv32 -filetype=asm -riscv-add-build-attributes \
# RUN:   -mattr=+experimental-y,+zce | FileCheck %s --check-prefixes=RV32Y-ZCE
# RUN: llvm-mc %s -triple=riscv32 -filetype=asm -riscv-add-build-attributes \
# RUN:   -mattr=+experimental-y,+zce,+f | FileCheck %s --check-prefixes=RV32Y-ZCE-F

## RV64Y + C + D: y is enabled, so zcd is not implied.
# RV64Y-CD: .attribute 5, "rv64i2p1_f2p2_d2p2_c2p0_y0p98_zicsr2p0_zca1p0"

## RV32Y + C + F: y is enabled, so zcf is not implied.
# RV32Y-CF: .attribute 5, "rv32i2p1_f2p2_c2p0_y0p98_zicsr2p0_zca1p0"

## RV32Y + ZCA + F: zca + y implies c on RV32 (y replaces zcf in the implication).
# RV32Y-ZCA-F: .attribute 5, "rv32i2p1_f2p2_c2p0_y0p98_zicsr2p0_zca1p0"

## RV64Y + ZCA + D: zca + y implies c on RV64 (y replaces zcd in the implication).
# RV64Y-ZCA-D: .attribute 5, "rv64i2p1_f2p2_d2p2_c2p0_y0p98_zicsr2p0_zca1p0"

## RV64Y + ZCA + ZCB: zca + zcb + y does NOT imply zce on RV64Y because Zce is incompatible. zca also implies c.
# RV64Y-ZCA-ZCB: .attribute 5, "rv64i2p1_c2p0_y0p98_zca1p0_zcb1p0"

## RV32Y + ZCE: zce implies zca, zcb, zcmp, zcmt. zca implies c.
# RV32Y-ZCE: .attribute 5, "rv32i2p1_c2p0_y0p98_zicsr2p0_zca1p0_zcb1p0_zce1p0_zcmp1p0_zcmt1p0"

## RV32Y + ZCE + F: y is enabled, so zcf is not implied by zce + f. zca + y + f implies c.
# RV32Y-ZCE-F: .attribute 5, "rv32i2p1_f2p2_c2p0_y0p98_zicsr2p0_zca1p0_zcb1p0_zce1p0_zcmp1p0_zcmt1p0"
