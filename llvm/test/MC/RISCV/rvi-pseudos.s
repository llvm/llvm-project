# RUN: llvm-mc %s -triple=riscv32 | FileCheck %s --check-prefixes=CHECK,CHECK-NOPIC
# RUN: llvm-mc %s -triple=riscv64 | FileCheck %s --check-prefixes=CHECK,CHECK-NOPIC
# RUN: llvm-mc %s -triple=riscv32 -position-independent \
# RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-PIC,CHECK-PIC-RV32
# RUN: llvm-mc %s -triple=riscv64 -position-independent \
# RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-PIC,CHECK-PIC-RV64

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

# CHECK: .Lpcrel_hi5:
# CHECK-NOPIC: auipc a0, %pcrel_hi(a_symbol)
# CHECK-NOPIC: addi  a0, a0, %pcrel_lo(.Lpcrel_hi5)
# CHECK-PIC:      auipc a0, %got_pcrel_hi(a_symbol)
# CHECK-PIC-RV32: lw    a0, %pcrel_lo(.Lpcrel_hi5)(a0)
# CHECK-PIC-RV64: ld    a0, %pcrel_lo(.Lpcrel_hi5)(a0)
la a0, a_symbol

# CHECK: .Lpcrel_hi6:
# CHECK-NOPIC: auipc a1, %pcrel_hi(another_symbol)
# CHECK-NOPIC: addi  a1, a1, %pcrel_lo(.Lpcrel_hi6)
# CHECK-PIC:      auipc a1, %got_pcrel_hi(another_symbol)
# CHECK-PIC-RV32: lw    a1, %pcrel_lo(.Lpcrel_hi6)(a1)
# CHECK-PIC-RV64: ld    a1, %pcrel_lo(.Lpcrel_hi6)(a1)
la a1, another_symbol

# Check that we can load the address of symbols that are spelled like a register
# CHECK: .Lpcrel_hi7:
# CHECK-NOPIC: auipc a2, %pcrel_hi(zero)
# CHECK-NOPIC: addi  a2, a2, %pcrel_lo(.Lpcrel_hi7)
# CHECK-PIC:      auipc a2, %got_pcrel_hi(zero)
# CHECK-PIC-RV32: lw    a2, %pcrel_lo(.Lpcrel_hi7)(a2)
# CHECK-PIC-RV64: ld    a2, %pcrel_lo(.Lpcrel_hi7)(a2)
la a2, zero

# CHECK: .Lpcrel_hi8:
# CHECK-NOPIC: auipc a3, %pcrel_hi(ra)
# CHECK-NOPIC: addi  a3, a3, %pcrel_lo(.Lpcrel_hi8)
# CHECK-PIC:      auipc a3, %got_pcrel_hi(ra)
# CHECK-PIC-RV32: lw    a3, %pcrel_lo(.Lpcrel_hi8)(a3)
# CHECK-PIC-RV64: ld    a3, %pcrel_lo(.Lpcrel_hi8)(a3)
la a3, ra

# CHECK: .Lpcrel_hi9:
# CHECK-NOPIC: auipc a4, %pcrel_hi(f1)
# CHECK-NOPIC: addi  a4, a4, %pcrel_lo(.Lpcrel_hi9)
# CHECK-PIC:      auipc a4, %got_pcrel_hi(f1)
# CHECK-PIC-RV32: lw    a4, %pcrel_lo(.Lpcrel_hi9)(a4)
# CHECK-PIC-RV64: ld    a4, %pcrel_lo(.Lpcrel_hi9)(a4)
la a4, f1

# CHECK: .Lpcrel_hi10:
# CHECK: auipc a0, %pcrel_hi(a_symbol)
# CHECK: lb  a0, %pcrel_lo(.Lpcrel_hi10)(a0)
lb a0, a_symbol

# CHECK: .Lpcrel_hi11:
# CHECK: auipc a1, %pcrel_hi(a_symbol)
# CHECK: lh  a1, %pcrel_lo(.Lpcrel_hi11)(a1)
lh a1, a_symbol

# CHECK: .Lpcrel_hi12:
# CHECK: auipc a2, %pcrel_hi(a_symbol)
# CHECK: lhu  a2, %pcrel_lo(.Lpcrel_hi12)(a2)
lhu a2, a_symbol

# CHECK: .Lpcrel_hi13:
# CHECK: auipc a3, %pcrel_hi(a_symbol)
# CHECK: lw  a3, %pcrel_lo(.Lpcrel_hi13)(a3)
lw a3, a_symbol

# CHECK: .Lpcrel_hi14:
# CHECK: auipc a4, %pcrel_hi(a_symbol)
# CHECK: sb  a3, %pcrel_lo(.Lpcrel_hi14)(a4)
sb a3, a_symbol, a4

# CHECK: .Lpcrel_hi15:
# CHECK: auipc a4, %pcrel_hi(a_symbol)
# CHECK: sh  a3, %pcrel_lo(.Lpcrel_hi15)(a4)
sh a3, a_symbol, a4

# CHECK: .Lpcrel_hi16:
# CHECK: auipc a4, %pcrel_hi(a_symbol)
# CHECK: sw  a3, %pcrel_lo(.Lpcrel_hi16)(a4)
sw a3, a_symbol, a4

# Check that we can load the address of symbols that are spelled like a register
# CHECK: .Lpcrel_hi17:
# CHECK: auipc a2, %pcrel_hi(zero)
# CHECK: lw  a2, %pcrel_lo(.Lpcrel_hi17)(a2)
lw a2, zero

# CHECK: .Lpcrel_hi18:
# CHECK: auipc a4, %pcrel_hi(zero)
# CHECK: sw  a3, %pcrel_lo(.Lpcrel_hi18)(a4)
sw a3, zero, a4
