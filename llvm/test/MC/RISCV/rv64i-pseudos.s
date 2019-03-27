# RUN: llvm-mc %s -triple=riscv64 | FileCheck %s

# CHECK: .Lpcrel_hi0:
# CHECK: auipc a2, %pcrel_hi(a_symbol)
# CHECK: lwu  a2, %pcrel_lo(.Lpcrel_hi0)(a2)
lwu a2, a_symbol

# CHECK: .Lpcrel_hi1:
# CHECK: auipc a3, %pcrel_hi(a_symbol)
# CHECK: ld  a3, %pcrel_lo(.Lpcrel_hi1)(a3)
ld a3, a_symbol

# CHECK: .Lpcrel_hi2:
# CHECK: auipc a4, %pcrel_hi(a_symbol)
# CHECK: sd  a3, %pcrel_lo(.Lpcrel_hi2)(a4)
sd a3, a_symbol, a4
