# Xqcilo - Qualcomm uC Large Offset Load Store extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcilo \
# RUN:     | FileCheck -check-prefixes=CHECK %s

# CHECK-LABEL: .Lpcrel_hi0
# CHECK-NEXT: auipc a0, %pcrel_hi(undefined)
# CHECK-NEXT: lb a0, %pcrel_lo(.Lpcrel_hi0)(a0)
qc.e.lb a0, undefined

# CHECK-LABEL: .Lpcrel_hi1
# CHECK-NEXT: auipc a0, %pcrel_hi(undefined)
# CHECK-NEXT: lbu a0, %pcrel_lo(.Lpcrel_hi1)(a0)
qc.e.lbu a0, undefined

# CHECK-LABEL: .Lpcrel_hi2
# CHECK-NEXT: auipc a0, %pcrel_hi(undefined)
# CHECK-NEXT: lh a0, %pcrel_lo(.Lpcrel_hi2)(a0)
qc.e.lh a0, undefined

# CHECK-LABEL: .Lpcrel_hi3
# CHECK-NEXT: auipc a0, %pcrel_hi(undefined)
# CHECK-NEXT: lhu a0, %pcrel_lo(.Lpcrel_hi3)(a0)
qc.e.lhu a0, undefined

# CHECK-LABEL: .Lpcrel_hi4
# CHECK-NEXT: auipc a0, %pcrel_hi(undefined)
# CHECK-NEXT: lw a0, %pcrel_lo(.Lpcrel_hi4)(a0)
qc.e.lw a0, undefined

# CHECK-LABEL: .Lpcrel_hi5
# CHECK-NEXT: auipc t0, %pcrel_hi(undefined)
# CHECK-NEXT: sb a0, %pcrel_lo(.Lpcrel_hi5)(t0)
qc.e.sb a0, undefined, t0

# CHECK-LABEL: .Lpcrel_hi6
# CHECK-NEXT: auipc t0, %pcrel_hi(undefined)
# CHECK-NEXT: sh a0, %pcrel_lo(.Lpcrel_hi6)(t0)
qc.e.sh a0, undefined, t0

# CHECK-LABEL: .Lpcrel_hi7
# CHECK-NEXT: auipc t0, %pcrel_hi(undefined)
# CHECK-NEXT: sw a0, %pcrel_lo(.Lpcrel_hi7)(t0)
qc.e.sw a0, undefined, t0
