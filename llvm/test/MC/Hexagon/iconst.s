# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s

a:
# CHECK: r0 = add(r0,#0)
# CHECK: R_HEX_27_REG
r0 = iconst(#a)
