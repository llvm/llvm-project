# RUN: llvm-mc %s -triple=riscv32 | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc %s -triple=riscv64 | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:   | llvm-objdump -d -riscv-no-aliases - | FileCheck -check-prefix=DISASM %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:   | llvm-objdump -d -riscv-no-aliases - | FileCheck -check-prefix=DISASM %s

# Checks change of options does not cause error: could not find corresponding %pcrel_hi 
# when assembling pseudoinstruction and its extended form.

.option push
.option norelax
  la a0, a_symbol
.option pop
  la a1, another_symbol

# ASM: .Lpcrel_hi0:
# ASM: auipc   a0, %pcrel_hi(a_symbol)
# ASM: addi    a0, a0, %pcrel_lo(.Lpcrel_hi0)
# ASM: .Lpcrel_hi1:
# ASM: auipc   a1, %pcrel_hi(another_symbol)
# ASM: addi    a1, a1, %pcrel_lo(.Lpcrel_hi1)

# DISASM: .Lpcrel_hi0:
# DISASM: auipc   a0, 0
# DISASM: addi    a0, a0, 0
# DISASM:.Lpcrel_hi1:
# DISASM: auipc   a1, 0
# DISASM: addi    a1, a1, 0

.option push
.option norelax
1:auipc   a0, %pcrel_hi(a_symbol)
  addi    a0, a0, %pcrel_lo(1b)
.option pop
2:auipc   a1, %pcrel_hi(another_symbol)
  addi    a1, a1, %pcrel_lo(2b)

# ASM: .Ltmp0:
# ASM: auipc   a0, %pcrel_hi(a_symbol)
# ASM: addi    a0, a0, %pcrel_lo(.Ltmp0)
# ASM: .Ltmp1:
# ASM: auipc   a1, %pcrel_hi(another_symbol)
# ASM: addi    a1, a1, %pcrel_lo(.Ltmp1)

# DISASM: .Ltmp0:
# DISASM: auipc   a0, 0
# DISASM: addi    a0, a0, 0
# DISASM: .Ltmp1:
# DISASM: auipc   a1, 0
# DISASM: addi    a1, a1, 0
