# RUN: llvm-mc -triple mips-unknown-linux-gnu -show-inst %s \
# RUN:   | FileCheck --check-prefix=STATIC %s
# RUN: llvm-mc -triple mips-unknown-linux-gnu -position-independent \
# RUN:         -show-inst %s | FileCheck --check-prefix=PIC %s
# RUN: llvm-mc -triple mips-unknown-linux-gnu -mattr=+micromips -show-inst %s \
# RUN:   | FileCheck --check-prefix=STATIC-MM %s
# RUN: llvm-mc -triple mips-unknown-linux-gnu -mattr=+micromips \
# RUN:         -position-independent -show-inst %s \
# RUN:   | FileCheck --check-prefix=PIC-MM %s

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
