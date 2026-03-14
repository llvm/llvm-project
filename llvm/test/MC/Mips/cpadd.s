# RUN: llvm-mc -triple=mips-unknown-linux-gnu -position-independent %s \
# RUN:   | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -triple=mips64-unknown-linux-gnu -position-independent %s \
# RUN:   | FileCheck -check-prefix=ASM %s
# RUN: llvm-mc -triple=mips-unknown-linux-gnu %s \
# RUN:   | FileCheck -check-prefix=ASM %s

# RUN: llvm-mc -triple=mips-unknown-linux-gnu \
# RUN:         -position-independent -filetype=obj -o - %s \
# RUN:   | llvm-objdump -d -r - | FileCheck -check-prefix=OBJ32-PIC %s
# RUN: llvm-mc -triple=mips64-unknown-linux-gnu \
# RUN:         -position-independent -filetype=obj -o - %s \
# RUN:   | llvm-objdump -d -r - | FileCheck -check-prefix=OBJ64-PIC %s

# RUN: llvm-mc -triple=mips-unknown-linux-gnu \
# RUN:         -filetype=obj -o - %s \
# RUN:   | llvm-objdump -d -r - | FileCheck -check-prefix=OBJ32-NPIC %s
# RUN: llvm-mc -triple=mips64-unknown-linux-gnu \
# RUN:         -filetype=obj -o - %s \
# RUN:   | llvm-objdump -d -r - | FileCheck -check-prefix=OBJ64-NPIC %s

# ASM:       .cpadd $4
# OBJ32-PIC: addu  $4, $4, $gp
# OBJ64-PIC: daddu $4, $4, $gp
# OBJ32-NPIC-NOT: addu
# OBJ64-NPIC-NOT: daddu

  .text
  .cpadd $4
