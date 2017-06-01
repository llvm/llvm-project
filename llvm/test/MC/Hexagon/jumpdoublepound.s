# RUN: llvm-mc -triple=hexagon -filetype=obj %s -o - | llvm-objdump -d - | FileCheck %s

# Verify that jump encodes correctly


mylabel:
# CHECK: if (p0) jump
if (p0) jump ##mylabel

# CHECK: if (cmp.gtu(r5.new, r4)) jump:t
{ r5 = r4
  if (cmp.gtu(r5.new, r4)) jump:t ##mylabel }

