# RUN: llvm-mc -arch=hexagon -filetype=obj %s | llvm-objdump -d - | FileCheck %s
#

# CHECK-NOT: immext
r0 = #undefined@TPREL

# CHECK-NOT: immext
r0 = #undefined@DTPREL
