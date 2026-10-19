# RUN: llvm-mc -triple=mips-unknown-linux -show-inst -mcpu=mips32r5 %s \
# RUN:   | FileCheck --check-prefix=R5 %s
# RUN: llvm-mc -triple=mips-unknown-linux -show-inst -mcpu=mips32r6 %s \
# RUN:   | FileCheck --check-prefix=R6 %s

  .text
  j $ra

# R5:    jr $ra
# R6:    jalr $ra
