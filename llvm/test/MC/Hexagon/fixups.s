# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

  .text
# CHECK-LABEL: 0:
# CHECK: 2442e106
# CHECK: if (!cmp.eq(r1.new,#1)) jump:t 0xc
  {
    r1 = zxth(r2)
    if (!cmp.eq(r1.new, #1)) jump:t .L1
  }
  nop
.L1:
  .org 0x10
# CHECK-LABEL: 10:
# CHECK: 00004020
# CHECK: immext(#2048)
# CHECK: 2442e118
# CHECK: if (!cmp.eq(r1.new,#1)) jump:t 0x81c
  {
    r1 = zxth(r2)
    if (!cmp.eq(r1.new, #1)) jump:t .L2
  }
  .org .+2048
.L2:

