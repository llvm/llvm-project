# RUN: llvm-mc -triple=mips64-unknown-linux-gnuabin32 -position-independent %s \
# RUN:   | FileCheck -check-prefix=ASM-PIC32 %s
# RUN: llvm-mc -triple=mips64-unknown-linux-gnu -position-independent %s \
# RUN:   | FileCheck -check-prefix=ASM-PIC64 %s
# RUN: llvm-mc -triple=mips64-unknown-linux-gnuabin32 %s \
# RUN:   | FileCheck -check-prefix=ASM-NPIC %s
# RUN: llvm-mc -triple=mips64-unknown-linux-gnu %s \
# RUN:   | FileCheck -check-prefix=ASM-NPIC %s

# RUN: llvm-mc -triple=mips64-unknown-linux-gnuabin32 \
# RUN:         -position-independent -filetype=obj -o - %s \
# RUN:   | llvm-objdump -d -r - | FileCheck --check-prefix=OBJ32 %s
# RUN: llvm-mc -triple=mips64-unknown-linux-gnu \
# RUN:         -position-independent -filetype=obj -o - %s \
# RUN:   | llvm-objdump -d -r - | FileCheck --check-prefix=OBJ64 %s

# ASM-PIC32:  .text
# ASM-PIC32:  .cplocal $4
# ASM-PIC32:  lw $25, %call16(foo)($4)
# ASM-PIC32:  jalr $25

# ASM-PIC64:  .text
# ASM-PIC64:  .cplocal $4
# ASM-PIC64:  ld $25, %call16(foo)($4)
# ASM-PIC64:  jalr $25

# ASM-NPIC:  .text
# ASM-NPIC:  .cplocal $4
# ASM-NPIC:  jal foo

# OBJ32:   lw $25, 0($4)
# OBJ32: R_MIPS_CALL16 foo
# OBJ32:   jalr $25
# OBJ32: R_MIPS_JALR foo

# OBJ64:   ld $25, 0($4)
# OBJ64: R_MIPS_CALL16/R_MIPS_NONE/R_MIPS_NONE foo
# OBJ64:   jalr $25
# OBJ64: R_MIPS_JALR/R_MIPS_NONE/R_MIPS_NONE foo

  .text
  .cplocal $4
  jal foo
foo:
  nop
