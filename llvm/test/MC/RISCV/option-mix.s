# RUN: llvm-mc %s -triple=riscv32 | FileCheck -check-prefixes=ASM %s
# RUN: llvm-mc %s -triple=riscv64 | FileCheck -check-prefixes=ASM %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+relax | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+relax | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:   | llvm-objdump -d -M no-aliases - | FileCheck -check-prefixes=DISASM,DISASM-NORELAX %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:   | llvm-objdump -d -M no-aliases - | FileCheck -check-prefixes=DISASM,DISASM-NORELAX %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+relax < %s \
# RUN:   | llvm-objdump -d -M no-aliases - | FileCheck -check-prefixes=DISASM,DISASM-RELAX %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+relax < %s \
# RUN:   | llvm-objdump -d -M no-aliases - | FileCheck -check-prefixes=DISASM,DISASM-RELAX %s

# Checks change of options does not cause error: could not find corresponding %pcrel_hi
# when assembling pseudoinstruction and its extended form. Also checks that we
# evaluate the correct value for local symbols in such a situation.

.option push
.option norelax
  la a0, a_symbol
.option pop
  la a1, another_symbol

# ASM-LABEL: .Lpcrel_hi0{{>?}}:
# ASM-NEXT: auipc   a0, %pcrel_hi(a_symbol)
# ASM-NEXT: addi    a0, a0, %pcrel_lo(.Lpcrel_hi0)
# ASM-LABEL: .Lpcrel_hi1{{>?}}:
# ASM-NEXT: auipc   a1, %pcrel_hi(another_symbol)
# ASM-NEXT: addi    a1, a1, %pcrel_lo(.Lpcrel_hi1)

# DISASM-LABEL: <.Lpcrel_hi0>:
# DISASM-NEXT: auipc   a0, 0
# DISASM-NEXT: addi    a0, a0, 0
# DISASM-LABEL: <.Lpcrel_hi1>:
# DISASM-NEXT: auipc   a1, 0
# DISASM-NEXT: addi    a1, a1, 0

.option push
.option norelax
1:auipc   a0, %pcrel_hi(a_symbol)
  addi    a0, a0, %pcrel_lo(1b)
.option pop
2:auipc   a1, %pcrel_hi(another_symbol)
  addi    a1, a1, %pcrel_lo(2b)

# ASM-LABEL: .Ltmp0{{>?}}:
# ASM-NEXT: auipc   a0, %pcrel_hi(a_symbol)
# ASM-NEXT: addi    a0, a0, %pcrel_lo(.Ltmp0)
# ASM-LABEL: .Ltmp1{{>?}}:
# ASM-NEXT: auipc   a1, %pcrel_hi(another_symbol)
# ASM-NEXT: addi    a1, a1, %pcrel_lo(.Ltmp1)

# DISASM-LABEL: .Ltmp0{{>?}}:
# DISASM-NEXT: auipc   a0, 0
# DISASM-NEXT: addi    a0, a0, 0
# DISASM-LABEL: .Ltmp1{{>?}}:
# DISASM-NEXT: auipc   a1, 0
# DISASM-NEXT: addi    a1, a1, 0

.option push
.option norelax
  la a0, a_symbol
.option pop
  la a1, local_symbol1

local_symbol1:
  nop

# ASM-LABEL: .Lpcrel_hi2{{>?}}:
# ASM-NEXT: auipc   a0, %pcrel_hi(a_symbol)
# ASM-NEXT: addi    a0, a0, %pcrel_lo(.Lpcrel_hi2)
# ASM-LABEL: .Lpcrel_hi3{{>?}}:
# ASM-NEXT: auipc   a1, %pcrel_hi(local_symbol1)
# ASM-NEXT: addi    a1, a1, %pcrel_lo(.Lpcrel_hi3)

# DISASM-LABEL: .Lpcrel_hi2{{>?}}:
# DISASM-NEXT: auipc   a0, 0
# DISASM-NEXT: addi    a0, a0, 0
# DISASM-NORELAX-NEXT: auipc   a1, 0
# DISASM-NORELAX-NEXT: addi    a1, a1, 8
# DISASM-RELAX-LABEL: .Lpcrel_hi3{{>?}}:
# DISASM-RELAX-NEXT: auipc   a1, 0
# DISASM-RELAX-NEXT: addi    a1, a1, 0

.option push
.option norelax
1:auipc   a0, %pcrel_hi(a_symbol)
  addi    a0, a0, %pcrel_lo(1b)
.option pop
2:auipc   a1, %pcrel_hi(local_symbol2)
  addi    a1, a1, %pcrel_lo(2b)

local_symbol2:
  nop

# ASM-LABEL: .Ltmp2{{>?}}:
# ASM-NEXT: auipc   a0, %pcrel_hi(a_symbol)
# ASM-NEXT: addi    a0, a0, %pcrel_lo(.Ltmp2)
# ASM-LABEL: .Ltmp3{{>?}}:
# ASM-NEXT: auipc   a1, %pcrel_hi(local_symbol2)
# ASM-NEXT: addi    a1, a1, %pcrel_lo(.Ltmp3)

# DISASM-LABEL: .Ltmp2{{>?}}:
# DISASM-NEXT: auipc   a0, 0
# DISASM-NEXT: addi    a0, a0, 0
# DISASM-NORELAX-NEXT: auipc   a1, 0
# DISASM-NORELAX-NEXT: addi    a1, a1, 8
# DISASM-RELAX-LABEL: .Ltmp3{{>?}}:
# DISASM-RELAX-NEXT: auipc   a1, 0
# DISASM-RELAX-NEXT: addi    a1, a1, 0
