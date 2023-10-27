# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump -d - | FileCheck %s
#

.I1:
nop

# CHECK: <.I1>:
# CHECK:        nop
