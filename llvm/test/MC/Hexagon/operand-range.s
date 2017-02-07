# RUN: not llvm-mc -arch=hexagon -filetype=asm %s 2>&1 | FileCheck %s

# Expect errors here, insn needs to be extended
R1 = mpyi(R2, #-256)
# CHECK: error:
R3 = mpyi(R4, #256)
# CHECK: error:
