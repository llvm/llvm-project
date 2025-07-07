# RUN: llvm-mc %s -triple=riscv32 -mattr=+zilsd | FileCheck %s

# CHECK: .Lpcrel_hi0:
# CHECK: auipc a4, %pcrel_hi(a_symbol)
# CHECK: ld  a4, %pcrel_lo(.Lpcrel_hi0)(a4)
ld a4, a_symbol

# CHECK: .Lpcrel_hi1:
# CHECK: auipc a3, %pcrel_hi(a_symbol)
# CHECK: sd  a6, %pcrel_lo(.Lpcrel_hi1)(a3)
sd a6, a_symbol, a3
