# RUN: llvm-mc %s -triple=riscv32 | FileCheck %s
# RUN: llvm-mc %s -triple=riscv64 | FileCheck %s

# CHECK: .Lpcrel_hi0:
# CHECK: auipc a0, %pcrel_hi(a_symbol)
# CHECK: addi  a0, a0, %pcrel_lo(.Lpcrel_hi0)
lla a0, a_symbol

# CHECK: .Lpcrel_hi1:
# CHECK: auipc a1, %pcrel_hi(another_symbol)
# CHECK: addi  a1, a1, %pcrel_lo(.Lpcrel_hi1)
lla a1, another_symbol

# Check that we can load the address of symbols that are spelled like a register
# CHECK: .Lpcrel_hi2:
# CHECK: auipc a2, %pcrel_hi(zero)
# CHECK: addi  a2, a2, %pcrel_lo(.Lpcrel_hi2)
lla a2, zero

# CHECK: .Lpcrel_hi3:
# CHECK: auipc a3, %pcrel_hi(ra)
# CHECK: addi  a3, a3, %pcrel_lo(.Lpcrel_hi3)
lla a3, ra

# CHECK: .Lpcrel_hi4:
# CHECK: auipc a4, %pcrel_hi(f1)
# CHECK: addi  a4, a4, %pcrel_lo(.Lpcrel_hi4)
lla a4, f1
