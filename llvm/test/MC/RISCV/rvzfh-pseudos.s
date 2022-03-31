# RUN: llvm-mc %s -triple=riscv32 -mattr=+zfh | FileCheck %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zfh | FileCheck %s

# CHECK: .Lpcrel_hi0:
# CHECK: auipc a2, %pcrel_hi(a_symbol)
# CHECK: flh  fa2, %pcrel_lo(.Lpcrel_hi0)(a2)
flh fa2, a_symbol, a2

# CHECK: .Lpcrel_hi1:
# CHECK: auipc a3, %pcrel_hi(a_symbol)
# CHECK: fsh  fa2, %pcrel_lo(.Lpcrel_hi1)(a3)
fsh fa2, a_symbol, a3
