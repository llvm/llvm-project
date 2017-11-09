# RUN: llvm-mc -arch=mips -show-inst %s | FileCheck --check-prefix=STATIC %s
# RUN: llvm-mc -arch=mips -position-independent -show-inst %s | FileCheck --check-prefix=PIC %s
# RUN: llvm-mc -arch=mips -mattr=+micromips -show-inst %s | FileCheck --check-prefix=STATIC-MM %s
# RUN: llvm-mc -arch=mips -mattr=+micromips -position-independent -show-inst %s | FileCheck --check-prefix=PIC-MM %s

  .text
  j foo
  nop
foo:
  nop

  b foo

# PIC:       b foo                   # <MCInst #{{[0-9]+}} BEQ
# STATIC:    j foo                   # <MCInst #{{[0-9]+}} J
# PIC-MM:    b foo                   # <MCInst #{{[0-9]+}} BEQ_MM
# STATIC-MM: j foo                   # <MCInst #{{[0-9]+}} J_MM
