# RUN: not llvm-mc -arch=hexagon -filetype=asm %s 2> %t; FileCheck %s < %t
#

# Check that changes to a read-only register is caught.

{ c9:8 = r1:0 }
# CHECK: error: Cannot write to read-only register
