# RUN: llvm-mc %s -triple=riscv32 \
# RUN:     | FileCheck %s --check-prefixes=CHECK,CHECK-NOPIC,CHECK-RV32
# RUN: llvm-mc %s -triple=riscv64 \
# RUN:     | FileCheck %s --check-prefixes=CHECK,CHECK-NOPIC,CHECK-RV64
# RUN: llvm-mc %s -triple=riscv32 -position-independent \
# RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-PIC,CHECK-RV32,CHECK-PIC-RV32
# RUN: llvm-mc %s -triple=riscv64 -position-independent \
# RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-PIC,CHECK-RV64,CHECK-PIC-RV64

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
# CHECK: auipc a5, %pcrel_hi(a_symbol+2040)
# CHECK: addi  a5, a5, %pcrel_lo(.Lpcrel_hi5)
lla a5, a_symbol + (0xFF << 3)

# CHECK: .Lpcrel_hi6:
# CHECK-NOPIC: auipc a0, %pcrel_hi(a_symbol)
# CHECK-NOPIC: addi  a0, a0, %pcrel_lo(.Lpcrel_hi6)
# CHECK-PIC:      auipc a0, %got_pcrel_hi(a_symbol)
# CHECK-PIC-RV32: lw    a0, %pcrel_lo(.Lpcrel_hi6)(a0)
# CHECK-PIC-RV64: ld    a0, %pcrel_lo(.Lpcrel_hi6)(a0)
la a0, a_symbol

# CHECK: .Lpcrel_hi7:
# CHECK-NOPIC: auipc a1, %pcrel_hi(another_symbol)
# CHECK-NOPIC: addi  a1, a1, %pcrel_lo(.Lpcrel_hi7)
# CHECK-PIC:      auipc a1, %got_pcrel_hi(another_symbol)
# CHECK-PIC-RV32: lw    a1, %pcrel_lo(.Lpcrel_hi7)(a1)
# CHECK-PIC-RV64: ld    a1, %pcrel_lo(.Lpcrel_hi7)(a1)
la a1, another_symbol

# Check that we can load the address of symbols that are spelled like a register
# CHECK: .Lpcrel_hi8:
# CHECK-NOPIC: auipc a2, %pcrel_hi(zero)
# CHECK-NOPIC: addi  a2, a2, %pcrel_lo(.Lpcrel_hi8)
# CHECK-PIC:      auipc a2, %got_pcrel_hi(zero)
# CHECK-PIC-RV32: lw    a2, %pcrel_lo(.Lpcrel_hi8)(a2)
# CHECK-PIC-RV64: ld    a2, %pcrel_lo(.Lpcrel_hi8)(a2)
la a2, zero

# CHECK: .Lpcrel_hi9:
# CHECK-NOPIC: auipc a3, %pcrel_hi(ra)
# CHECK-NOPIC: addi  a3, a3, %pcrel_lo(.Lpcrel_hi9)
# CHECK-PIC:      auipc a3, %got_pcrel_hi(ra)
# CHECK-PIC-RV32: lw    a3, %pcrel_lo(.Lpcrel_hi9)(a3)
# CHECK-PIC-RV64: ld    a3, %pcrel_lo(.Lpcrel_hi9)(a3)
la a3, ra

# CHECK: .Lpcrel_hi10:
# CHECK-NOPIC: auipc a4, %pcrel_hi(f1)
# CHECK-NOPIC: addi  a4, a4, %pcrel_lo(.Lpcrel_hi10)
# CHECK-PIC:      auipc a4, %got_pcrel_hi(f1)
# CHECK-PIC-RV32: lw    a4, %pcrel_lo(.Lpcrel_hi10)(a4)
# CHECK-PIC-RV64: ld    a4, %pcrel_lo(.Lpcrel_hi10)(a4)
la a4, f1

# CHECK: .Lpcrel_hi11:
# CHECK: auipc a0, %tls_ie_pcrel_hi(a_symbol)
# CHECK-RV32: lw    a0, %pcrel_lo(.Lpcrel_hi11)(a0)
# CHECK-RV64: ld    a0, %pcrel_lo(.Lpcrel_hi11)(a0)
la.tls.ie a0, a_symbol

# CHECK: .Lpcrel_hi12:
# CHECK: auipc a1, %tls_ie_pcrel_hi(another_symbol)
# CHECK-RV32: lw    a1, %pcrel_lo(.Lpcrel_hi12)(a1)
# CHECK-RV64: ld    a1, %pcrel_lo(.Lpcrel_hi12)(a1)
la.tls.ie a1, another_symbol

# Check that we can load the address of symbols that are spelled like a register
# CHECK: .Lpcrel_hi13:
# CHECK: auipc a2, %tls_ie_pcrel_hi(zero)
# CHECK-RV32: lw    a2, %pcrel_lo(.Lpcrel_hi13)(a2)
# CHECK-RV64: ld    a2, %pcrel_lo(.Lpcrel_hi13)(a2)
la.tls.ie a2, zero

# CHECK: .Lpcrel_hi14:
# CHECK: auipc a3, %tls_ie_pcrel_hi(ra)
# CHECK-RV32: lw    a3, %pcrel_lo(.Lpcrel_hi14)(a3)
# CHECK-RV64: ld    a3, %pcrel_lo(.Lpcrel_hi14)(a3)
la.tls.ie a3, ra

# CHECK: .Lpcrel_hi15:
# CHECK: auipc a4, %tls_ie_pcrel_hi(f1)
# CHECK-RV32: lw    a4, %pcrel_lo(.Lpcrel_hi15)(a4)
# CHECK-RV64: ld    a4, %pcrel_lo(.Lpcrel_hi15)(a4)
la.tls.ie a4, f1

# CHECK: .Lpcrel_hi16:
# CHECK: auipc a0, %tls_gd_pcrel_hi(a_symbol)
# CHECK: addi  a0, a0, %pcrel_lo(.Lpcrel_hi16)
la.tls.gd a0, a_symbol

# CHECK: .Lpcrel_hi17:
# CHECK: auipc a1, %tls_gd_pcrel_hi(another_symbol)
# CHECK: addi  a1, a1, %pcrel_lo(.Lpcrel_hi17)
la.tls.gd a1, another_symbol

# Check that we can load the address of symbols that are spelled like a register
# CHECK: .Lpcrel_hi18:
# CHECK: auipc a2, %tls_gd_pcrel_hi(zero)
# CHECK: addi  a2, a2, %pcrel_lo(.Lpcrel_hi18)
la.tls.gd a2, zero

# CHECK: .Lpcrel_hi19:
# CHECK: auipc a3, %tls_gd_pcrel_hi(ra)
# CHECK: addi  a3, a3, %pcrel_lo(.Lpcrel_hi19)
la.tls.gd a3, ra

# CHECK: .Lpcrel_hi20:
# CHECK: auipc a4, %tls_gd_pcrel_hi(f1)
# CHECK: addi  a4, a4, %pcrel_lo(.Lpcrel_hi20)
la.tls.gd a4, f1

# CHECK: .Lpcrel_hi21:
# CHECK: auipc a0, %pcrel_hi(a_symbol)
# CHECK: lb  a0, %pcrel_lo(.Lpcrel_hi21)(a0)
lb a0, a_symbol

# CHECK: .Lpcrel_hi22:
# CHECK: auipc a1, %pcrel_hi(a_symbol)
# CHECK: lh  a1, %pcrel_lo(.Lpcrel_hi22)(a1)
lh a1, a_symbol

# CHECK: .Lpcrel_hi23:
# CHECK: auipc a2, %pcrel_hi(a_symbol)
# CHECK: lhu  a2, %pcrel_lo(.Lpcrel_hi23)(a2)
lhu a2, a_symbol

# CHECK: .Lpcrel_hi24:
# CHECK: auipc a3, %pcrel_hi(a_symbol)
# CHECK: lw  a3, %pcrel_lo(.Lpcrel_hi24)(a3)
lw a3, a_symbol

# CHECK: .Lpcrel_hi25:
# CHECK: auipc a4, %pcrel_hi(a_symbol)
# CHECK: sb  a3, %pcrel_lo(.Lpcrel_hi25)(a4)
sb a3, a_symbol, a4

# CHECK: .Lpcrel_hi26:
# CHECK: auipc a4, %pcrel_hi(a_symbol)
# CHECK: sh  a3, %pcrel_lo(.Lpcrel_hi26)(a4)
sh a3, a_symbol, a4

# CHECK: .Lpcrel_hi27:
# CHECK: auipc a4, %pcrel_hi(a_symbol)
# CHECK: sw  a3, %pcrel_lo(.Lpcrel_hi27)(a4)
sw a3, a_symbol, a4

# Check that we can load the address of symbols that are spelled like a register
# CHECK: .Lpcrel_hi28:
# CHECK: auipc a2, %pcrel_hi(zero)
# CHECK: lw  a2, %pcrel_lo(.Lpcrel_hi28)(a2)
lw a2, zero

# CHECK: .Lpcrel_hi29:
# CHECK: auipc a4, %pcrel_hi(zero)
# CHECK: sw  a3, %pcrel_lo(.Lpcrel_hi29)(a4)
sw a3, zero, a4

## Check that a complex expression can be simplified and matched.
# CHECK: .Lpcrel_hi30:
# CHECK: auipc a5, %pcrel_hi((255+a_symbol)-4)
# CHECK: addi  a5, a5, %pcrel_lo(.Lpcrel_hi30)
lla a5, (0xFF + a_symbol) - 4

## Check that we don't double-parse a top-level minus.
# CHECK: .Lpcrel_hi31:
# CHECK: auipc a5, %pcrel_hi(a_symbol-4)
# CHECK: addi  a5, a5, %pcrel_lo(.Lpcrel_hi31)
lla a5, a_symbol - 4
