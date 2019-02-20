# RUN: llvm-mc %s -triple=riscv32 -mattr=+f | FileCheck %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+f | FileCheck %s

# CHECK: .Lpcrel_hi0:
# CHECK: auipc a2, %pcrel_hi(a_symbol)
# CHECK: flw  fa2, %pcrel_lo(.Lpcrel_hi0)(a2)
flw fa2, a_symbol, a2

# CHECK: .Lpcrel_hi1:
# CHECK: auipc a3, %pcrel_hi(a_symbol)
# CHECK: fsw  fa2, %pcrel_lo(.Lpcrel_hi1)(a3)
fsw fa2, a_symbol, a3
