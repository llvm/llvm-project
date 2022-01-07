# RUN: not llvm-mc -arch=hexagon -filetype=asm %s 2>%t; FileCheck %s <%t

{ nop }:junk
# CHECK: 3:9: error: 'junk' is not a valid bundle option
