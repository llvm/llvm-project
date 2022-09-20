;; Generate ELF attributes from llc.

; RUN: llc -mtriple=riscv32 -mattr=+m %s -o - | FileCheck --check-prefix=RV32M %s
; RUN: llc -mtriple=riscv32 -mattr=+zmmul %s -o - | FileCheck --check-prefix=RV32ZMMUL %s
; RUN: llc -mtriple=riscv32 -mattr=+m,+zmmul %s -o - | FileCheck --check-prefix=RV32MZMMUL %s
; RUN: llc -mtriple=riscv32 -mattr=+a %s -o - | FileCheck --check-prefix=RV32A %s
; RUN: llc -mtriple=riscv32 -mattr=+f %s -o - | FileCheck --check-prefix=RV32F %s
; RUN: llc -mtriple=riscv32 -mattr=+d %s -o - | FileCheck --check-prefix=RV32D %s
; RUN: llc -mtriple=riscv32 -mattr=+c %s -o - | FileCheck --check-prefix=RV32C %s
; RUN: llc -mtriple=riscv32 -mattr=+zihintpause %s -o - | FileCheck --check-prefix=RV32ZIHINTPAUSE %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zihintntl %s -o - | FileCheck --check-prefix=RV32ZIHINTNTL %s
; RUN: llc -mtriple=riscv32 -mattr=+zfhmin %s -o - | FileCheck --check-prefix=RV32ZFHMIN %s
; RUN: llc -mtriple=riscv32 -mattr=+zfh %s -o - | FileCheck --check-prefix=RV32ZFH %s
; RUN: llc -mtriple=riscv32 -mattr=+zba %s -o - | FileCheck --check-prefix=RV32ZBA %s
; RUN: llc -mtriple=riscv32 -mattr=+zbb %s -o - | FileCheck --check-prefix=RV32ZBB %s
; RUN: llc -mtriple=riscv32 -mattr=+zbc %s -o - | FileCheck --check-prefix=RV32ZBC %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zbe %s -o - | FileCheck --check-prefix=RV32ZBE %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zbf %s -o - | FileCheck --check-prefix=RV32ZBF %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zbm %s -o - | FileCheck --check-prefix=RV32ZBM %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zbp %s -o - | FileCheck --check-prefix=RV32ZBP %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zbr %s -o - | FileCheck --check-prefix=RV32ZBR %s
; RUN: llc -mtriple=riscv32 -mattr=+zbs %s -o - | FileCheck --check-prefix=RV32ZBS %s
; RUN: llc -mtriple=riscv32 -mattr=+experimental-zbt %s -o - | FileCheck --check-prefix=RV32ZBT %s
; RUN: llc -mtriple=riscv32 -mattr=+v %s -o - | FileCheck --check-prefix=RV32V %s
; RUN: llc -mtriple=riscv32 -mattr=+zbb,+zfh,+v,+f %s -o - | FileCheck --check-prefix=RV32COMBINED %s
; RUN: llc -mtriple=riscv32 -mattr=+zbkb %s -o - | FileCheck --check-prefix=RV32ZBKB %s
; RUN: llc -mtriple=riscv32 -mattr=+zbkc %s -o - | FileCheck --check-prefix=RV32ZBKC %s
; RUN: llc -mtriple=riscv32 -mattr=+zbkx %s -o - | FileCheck --check-prefix=RV32ZBKX %s
; RUN: llc -mtriple=riscv32 -mattr=+zknd %s -o - | FileCheck --check-prefix=RV32ZKND %s
; RUN: llc -mtriple=riscv32 -mattr=+zkne %s -o - | FileCheck --check-prefix=RV32ZKNE %s
; RUN: llc -mtriple=riscv32 -mattr=+zknh %s -o - | FileCheck --check-prefix=RV32ZKNH %s
; RUN: llc -mtriple=riscv32 -mattr=+zksed %s -o - | FileCheck --check-prefix=RV32ZKSED %s
; RUN: llc -mtriple=riscv32 -mattr=+zksh %s -o - | FileCheck --check-prefix=RV32ZKSH %s
; RUN: llc -mtriple=riscv32 -mattr=+zkr %s -o - | FileCheck --check-prefix=RV32ZKR %s
; RUN: llc -mtriple=riscv32 -mattr=+zkn %s -o - | FileCheck --check-prefix=RV32ZKN %s
; RUN: llc -mtriple=riscv32 -mattr=+zks %s -o - | FileCheck --check-prefix=RV32ZKS %s
; RUN: llc -mtriple=riscv32 -mattr=+zkt %s -o - | FileCheck --check-prefix=RV32ZKT %s
; RUN: llc -mtriple=riscv32 -mattr=+zk %s -o - | FileCheck --check-prefix=RV32ZK %s
; RUN: llc -mtriple=riscv32 -mattr=+zkn,+zkr,+zkt %s -o - | FileCheck --check-prefix=RV32COMBINEINTOZK %s
; RUN: llc -mtriple=riscv32 -mattr=+zbkb,+zbkc,+zbkx,+zkne,+zknd,+zknh %s -o - | FileCheck --check-prefix=RV32COMBINEINTOZKN %s
; RUN: llc -mtriple=riscv32 -mattr=+zbkb,+zbkc,+zbkx,+zksed,+zksh %s -o - | FileCheck --check-prefix=RV32COMBINEINTOZKS %s
; RUN: llc -mtriple=riscv32 -mattr=+zicbom %s -o - | FileCheck --check-prefix=RV32ZICBOM %s
; RUN: llc -mtriple=riscv32 -mattr=+zicboz %s -o - | FileCheck --check-prefix=RV32ZICBOZ %s
; RUN: llc -mtriple=riscv32 -mattr=+zicbop %s -o - | FileCheck --check-prefix=RV32ZICBOP %s
; RUN: llc -mtriple=riscv64 -mattr=+m %s -o - | FileCheck --check-prefix=RV64M %s
; RUN: llc -mtriple=riscv64 -mattr=+zmmul %s -o - | FileCheck --check-prefix=RV64ZMMUL %s
; RUN: llc -mtriple=riscv64 -mattr=+m,+zmmul %s -o - | FileCheck --check-prefix=RV64MZMMUL %s
; RUN: llc -mtriple=riscv64 -mattr=+a %s -o - | FileCheck --check-prefix=RV64A %s
; RUN: llc -mtriple=riscv64 -mattr=+f %s -o - | FileCheck --check-prefix=RV64F %s
; RUN: llc -mtriple=riscv64 -mattr=+d %s -o - | FileCheck --check-prefix=RV64D %s
; RUN: llc -mtriple=riscv64 -mattr=+c %s -o - | FileCheck --check-prefix=RV64C %s
; RUN: llc -mtriple=riscv64 -mattr=+zihintpause %s -o - | FileCheck --check-prefix=RV64ZIHINTPAUSE %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zihintntl %s -o - | FileCheck --check-prefix=RV64ZIHINTNTL %s
; RUN: llc -mtriple=riscv64 -mattr=+zfhmin %s -o - | FileCheck --check-prefix=RV64ZFHMIN %s
; RUN: llc -mtriple=riscv64 -mattr=+zfh %s -o - | FileCheck --check-prefix=RV64ZFH %s
; RUN: llc -mtriple=riscv64 -mattr=+zba %s -o - | FileCheck --check-prefix=RV64ZBA %s
; RUN: llc -mtriple=riscv64 -mattr=+zbb %s -o - | FileCheck --check-prefix=RV64ZBB %s
; RUN: llc -mtriple=riscv64 -mattr=+zbc %s -o - | FileCheck --check-prefix=RV64ZBC %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zbe %s -o - | FileCheck --check-prefix=RV64ZBE %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zbf %s -o - | FileCheck --check-prefix=RV64ZBF %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zbm %s -o - | FileCheck --check-prefix=RV64ZBM %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zbp %s -o - | FileCheck --check-prefix=RV64ZBP %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zbr %s -o - | FileCheck --check-prefix=RV64ZBR %s
; RUN: llc -mtriple=riscv64 -mattr=+zbs %s -o - | FileCheck --check-prefix=RV64ZBS %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zbt %s -o - | FileCheck --check-prefix=RV64ZBT %s
; RUN: llc -mtriple=riscv64 -mattr=+v %s -o - | FileCheck --check-prefix=RV64V %s
; RUN: llc -mtriple=riscv64 -mattr=+zbb,+zfh,+v,+f %s -o - | FileCheck --check-prefix=RV64COMBINED %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkb %s -o - | FileCheck --check-prefix=RV64ZBKB %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkc %s -o - | FileCheck --check-prefix=RV64ZBKC %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkx %s -o - | FileCheck --check-prefix=RV64ZBKX %s
; RUN: llc -mtriple=riscv64 -mattr=+zknd %s -o - | FileCheck --check-prefix=RV64ZKND %s
; RUN: llc -mtriple=riscv64 -mattr=+zkne %s -o - | FileCheck --check-prefix=RV64ZKNE %s
; RUN: llc -mtriple=riscv64 -mattr=+zknh %s -o - | FileCheck --check-prefix=RV64ZKNH %s
; RUN: llc -mtriple=riscv64 -mattr=+zksed %s -o - | FileCheck --check-prefix=RV64ZKSED %s
; RUN: llc -mtriple=riscv64 -mattr=+zksh %s -o - | FileCheck --check-prefix=RV64ZKSH %s
; RUN: llc -mtriple=riscv64 -mattr=+zkr %s -o - | FileCheck --check-prefix=RV64ZKR %s
; RUN: llc -mtriple=riscv64 -mattr=+zkn %s -o - | FileCheck --check-prefix=RV64ZKN %s
; RUN: llc -mtriple=riscv64 -mattr=+zks %s -o - | FileCheck --check-prefix=RV64ZKS %s
; RUN: llc -mtriple=riscv64 -mattr=+zkt %s -o - | FileCheck --check-prefix=RV64ZKT %s
; RUN: llc -mtriple=riscv64 -mattr=+zk %s -o - | FileCheck --check-prefix=RV64ZK %s
; RUN: llc -mtriple=riscv64 -mattr=+zkn,+zkr,+zkt %s -o - | FileCheck --check-prefix=RV64COMBINEINTOZK %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkb,+zbkc,+zbkx,+zkne,+zknd,+zknh %s -o - | FileCheck --check-prefix=RV64COMBINEINTOZKN %s
; RUN: llc -mtriple=riscv64 -mattr=+zbkb,+zbkc,+zbkx,+zksed,+zksh %s -o - | FileCheck --check-prefix=RV64COMBINEINTOZKS %s
; RUN: llc -mtriple=riscv64 -mattr=+zicbom %s -o - | FileCheck --check-prefix=RV64ZICBOM %s
; RUN: llc -mtriple=riscv64 -mattr=+zicboz %s -o - | FileCheck --check-prefix=RV64ZICBOZ %s
; RUN: llc -mtriple=riscv64 -mattr=+zicbop %s -o - | FileCheck --check-prefix=RV64ZICBOP %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-zawrs %s -o - | FileCheck --check-prefix=RV64ZAWRS %s
; RUN: llc -mtriple=riscv64 -mattr=+experimental-ztso %s -o - | FileCheck --check-prefix=RV64ZTSO %s

; RV32M: .attribute 5, "rv32i2p0_m2p0"
; RV32ZMMUL: .attribute 5, "rv32i2p0_zmmul1p0"
; RV32MZMMUL: .attribute 5, "rv32i2p0_m2p0_zmmul1p0"
; RV32A: .attribute 5, "rv32i2p0_a2p0"
; RV32F: .attribute 5, "rv32i2p0_f2p0"
; RV32D: .attribute 5, "rv32i2p0_f2p0_d2p0"
; RV32C: .attribute 5, "rv32i2p0_c2p0"
; RV32ZIHINTPAUSE: .attribute 5, "rv32i2p0_zihintpause2p0"
; RV32ZIHINTNTL: .attribute 5, "rv32i2p0_zihintntl0p2"
; RV32ZFHMIN: .attribute 5, "rv32i2p0_f2p0_zfhmin1p0"
; RV32ZFH: .attribute 5, "rv32i2p0_f2p0_zfh1p0"
; RV32ZBA: .attribute 5, "rv32i2p0_zba1p0"
; RV32ZBB: .attribute 5, "rv32i2p0_zbb1p0"
; RV32ZBC: .attribute 5, "rv32i2p0_zbc1p0"
; RV32ZBE: .attribute 5, "rv32i2p0_zbe0p93"
; RV32ZBF: .attribute 5, "rv32i2p0_zbf0p93"
; RV32ZBM: .attribute 5, "rv32i2p0_zbm0p93"
; RV32ZBP: .attribute 5, "rv32i2p0_zbp0p93"
; RV32ZBR: .attribute 5, "rv32i2p0_zbr0p93"
; RV32ZBS: .attribute 5, "rv32i2p0_zbs1p0"
; RV32ZBT: .attribute 5, "rv32i2p0_zbt0p93"
; RV32V: .attribute 5, "rv32i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
; RV32COMBINED: .attribute 5, "rv32i2p0_f2p0_d2p0_v1p0_zfh1p0_zbb1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
; RV32ZBKB: .attribute 5, "rv32i2p0_zbkb1p0"
; RV32ZBKC: .attribute 5, "rv32i2p0_zbkc1p0"
; RV32ZBKX: .attribute 5, "rv32i2p0_zbkx1p0"
; RV32ZKND: .attribute 5, "rv32i2p0_zknd1p0"
; RV32ZKNE: .attribute 5, "rv32i2p0_zkne1p0"
; RV32ZKNH: .attribute 5, "rv32i2p0_zknh1p0"
; RV32ZKSED: .attribute 5, "rv32i2p0_zksed1p0"
; RV32ZKSH: .attribute 5, "rv32i2p0_zksh1p0"
; RV32ZKR: .attribute 5, "rv32i2p0_zkr1p0"
; RV32ZKN: .attribute 5, "rv32i2p0_zbkb1p0_zbkc1p0_zbkx1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0"
; RV32ZKS: .attribute 5, "rv32i2p0_zbkb1p0_zbkc1p0_zbkx1p0_zks1p0_zksed1p0_zksh1p0"
; RV32ZKT: .attribute 5, "rv32i2p0_zkt1p0"
; RV32ZK: .attribute 5, "rv32i2p0_zbkb1p0_zbkc1p0_zbkx1p0_zk1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0_zkr1p0_zkt1p0"
; RV32COMBINEINTOZK: .attribute 5, "rv32i2p0_zbkb1p0_zbkc1p0_zbkx1p0_zk1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0_zkr1p0_zkt1p0"
; RV32COMBINEINTOZKN: .attribute 5, "rv32i2p0_zbkb1p0_zbkc1p0_zbkx1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0"
; RV32COMBINEINTOZKS: .attribute 5, "rv32i2p0_zbkb1p0_zbkc1p0_zbkx1p0_zks1p0_zksed1p0_zksh1p0"
; RV32ZICBOM: .attribute 5, "rv32i2p0_zicbom1p0"
; RV32ZICBOZ: .attribute 5, "rv32i2p0_zicboz1p0"
; RV32ZICBOP: .attribute 5, "rv32i2p0_zicbop1p0"

; RV64M: .attribute 5, "rv64i2p0_m2p0"
; RV64ZMMUL: .attribute 5, "rv64i2p0_zmmul1p0"
; RV64MZMMUL: .attribute 5, "rv64i2p0_m2p0_zmmul1p0"
; RV64A: .attribute 5, "rv64i2p0_a2p0"
; RV64F: .attribute 5, "rv64i2p0_f2p0"
; RV64D: .attribute 5, "rv64i2p0_f2p0_d2p0"
; RV64C: .attribute 5, "rv64i2p0_c2p0"
; RV64ZIHINTPAUSE: .attribute 5, "rv64i2p0_zihintpause2p0"
; RV64ZIHINTNTL: .attribute 5, "rv64i2p0_zihintntl0p2"
; RV64ZFHMIN: .attribute 5, "rv64i2p0_f2p0_zfhmin1p0"
; RV64ZFH: .attribute 5, "rv64i2p0_f2p0_zfh1p0"
; RV64ZBA: .attribute 5, "rv64i2p0_zba1p0"
; RV64ZBB: .attribute 5, "rv64i2p0_zbb1p0"
; RV64ZBC: .attribute 5, "rv64i2p0_zbc1p0"
; RV64ZBE: .attribute 5, "rv64i2p0_zbe0p93"
; RV64ZBF: .attribute 5, "rv64i2p0_zbf0p93"
; RV64ZBM: .attribute 5, "rv64i2p0_zbm0p93"
; RV64ZBP: .attribute 5, "rv64i2p0_zbp0p93"
; RV64ZBR: .attribute 5, "rv64i2p0_zbr0p93"
; RV64ZBS: .attribute 5, "rv64i2p0_zbs1p0"
; RV64ZBT: .attribute 5, "rv64i2p0_zbt0p93"
; RV64V: .attribute 5, "rv64i2p0_f2p0_d2p0_v1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
; RV64COMBINED: .attribute 5, "rv64i2p0_f2p0_d2p0_v1p0_zfh1p0_zbb1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
; RV64ZBKB: .attribute 5, "rv64i2p0_zbkb1p0"
; RV64ZBKC: .attribute 5, "rv64i2p0_zbkc1p0"
; RV64ZBKX: .attribute 5, "rv64i2p0_zbkx1p0"
; RV64ZKND: .attribute 5, "rv64i2p0_zknd1p0"
; RV64ZKNE: .attribute 5, "rv64i2p0_zkne1p0"
; RV64ZKNH: .attribute 5, "rv64i2p0_zknh1p0"
; RV64ZKSED: .attribute 5, "rv64i2p0_zksed1p0"
; RV64ZKSH: .attribute 5, "rv64i2p0_zksh1p0"
; RV64ZKR: .attribute 5, "rv64i2p0_zkr1p0"
; RV64ZKN: .attribute 5, "rv64i2p0_zbkb1p0_zbkc1p0_zbkx1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0"
; RV64ZKS: .attribute 5, "rv64i2p0_zbkb1p0_zbkc1p0_zbkx1p0_zks1p0_zksed1p0_zksh1p0"
; RV64ZKT: .attribute 5, "rv64i2p0_zkt1p0"
; RV64ZK: .attribute 5, "rv64i2p0_zbkb1p0_zbkc1p0_zbkx1p0_zk1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0_zkr1p0_zkt1p0"
; RV64COMBINEINTOZK: .attribute 5, "rv64i2p0_zbkb1p0_zbkc1p0_zbkx1p0_zk1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0_zkr1p0_zkt1p0"
; RV64COMBINEINTOZKN: .attribute 5, "rv64i2p0_zbkb1p0_zbkc1p0_zbkx1p0_zkn1p0_zknd1p0_zkne1p0_zknh1p0"
; RV64COMBINEINTOZKS: .attribute 5, "rv64i2p0_zbkb1p0_zbkc1p0_zbkx1p0_zks1p0_zksed1p0_zksh1p0"
; RV64ZICBOM: .attribute 5, "rv64i2p0_zicbom1p0"
; RV64ZICBOZ: .attribute 5, "rv64i2p0_zicboz1p0"
; RV64ZAWRS: .attribute 5, "rv64i2p0_zawrs1p0"
; RV64ZICBOP: .attribute 5, "rv64i2p0_zicbop1p0"
; RV64ZTSO: .attribute 5, "rv64i2p0_ztso0p1"

define i32 @addi(i32 %a) {
  %1 = add i32 %a, 1
  ret i32 %1
}
