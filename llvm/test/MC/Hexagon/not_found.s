# RUN: not llvm-mc -arch=hexagon -filetype=asm junk123.s 2>%t ; FileCheck %s < %t
#

# CHECK: junk123.s: {{[N|n]}}o such file or directory
