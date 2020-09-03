# RUN: llvm-mc -arch=hexagon -filetype=obj %s | llvm-objdump -d - | FileCheck %s
#

.I1:
nop

# CHECK: <.I1>:
# CHECK:        nop
