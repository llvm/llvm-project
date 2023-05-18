# RUN: not llvm-mc -triple=hexagon -filetype=asm junk123.s 2>%t ; FileCheck -DMSG=%errc_ENOENT %s < %t
#

# CHECK: junk123.s: [[MSG]]
