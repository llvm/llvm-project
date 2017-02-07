# RUN: not llvm-mc -arch=hexagon %s 2> %t
# RUN: FileCheck < %t %s

.equ   a, 0
.set   a, 1
.equ   a, 2
.equiv a, 3
# CHECK: {{[Ee]}}rror: redefinition of 'a'

