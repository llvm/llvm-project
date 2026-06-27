# RUN: llvm-mc %s -triple=riscv32 --mattr=+xcvelw | FileCheck %s

# CHECK: .Lpcrel_hi0:
# CHECK: auipc a2, %pcrel_hi(a_symbol)
# CHECK: cv.elw a2, %pcrel_lo(.Lpcrel_hi0)(a2)
cv.elw a2, a_symbol

# CHECK: .Lpcrel_hi1:
# CHECK: auipc a3, %pcrel_hi(a_symbol)
# CHECK: cv.elw a3, %pcrel_lo(.Lpcrel_hi1)(a3)
cv.elw a3, a_symbol
