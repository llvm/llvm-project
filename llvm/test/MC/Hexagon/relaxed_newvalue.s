# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump --no-print-imm-hex -d - | FileCheck %s
# Make sure relaxation doesn't hinder newvalue calculation

#CHECK: r18 = add(r2,#-6)
#CHECK-NEXT: immext(#0)
#CHECK-NEXT: if (!cmp.gt(r18.new,#1)) jump:t
{
  r18 = add(r2, #-6)
  if (!cmp.gt(r18.new, #1)) jump:t .unknown
}
