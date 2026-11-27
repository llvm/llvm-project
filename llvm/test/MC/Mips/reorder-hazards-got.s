# RUN: llvm-mc -triple=mips -mcpu=mips1 -filetype=obj %s \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefixes=ALL,MIPS1
# RUN: llvm-mc -triple=mips -mcpu=mips2 -filetype=obj %s \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s --check-prefix=ALL

# A GOT load of a symbol that is not defined yet is expanded once the symbol
# is known to be local or global. On MIPS I, the addiu of the local sequence
# needs a nop after the load.

  .option pic2

# ALL-LABEL: <f>:
# ALL-NEXT:    lw $4, 0x0($gp)
# MIPS1-NEXT:  nop
# ALL-NEXT:    addiu $4, $4, 0x{{[0-9a-f]+}}
# MIPS1-NEXT:  nop
# ALL-NEXT:    addu $5, $4, $4
# ALL-NEXT:    lw $4, 0x0($gp)
# MIPS1-NEXT:  nop
# ALL-NEXT:    addu $5, $4, $4
# ALL-NEXT:    lw $4, 0x0($gp)
# MIPS1-NEXT:  nop
# ALL-NEXT:    addiu $4, $4, 0x{{[0-9a-f]+}}
# ALL-NEXT:    addu $5, $4, $4
f:
  la $4, later
  addu $5, $4, $4
  la $4, ext
  addu $5, $4, $4
  la $4, later+8
  addu $5, $4, $4
later:
  .word 0
