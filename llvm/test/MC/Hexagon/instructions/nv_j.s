# RUN: llvm-mc -triple=hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.7.1 NV/J

# Jump to address conditioned on new register value
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 d5 02 20
{ r17 = r17
  if (cmp.eq(r17.new, r21)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 f5 02 20
{ r17 = r17
  if (cmp.eq(r17.new, r21)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 d5 42 20
{ r17 = r17
  if (!cmp.eq(r17.new, r21)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 f5 42 20
{ r17 = r17
  if (!cmp.eq(r17.new, r21)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 d5 82 20
{ r17 = r17
  if (cmp.gt(r17.new, r21)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 f5 82 20
{ r17 = r17
  if (cmp.gt(r17.new, r21)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 d5 c2 20
{ r17 = r17
  if (!cmp.gt(r17.new, r21)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 f5 c2 20
{ r17 = r17
  if (!cmp.gt(r17.new, r21)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 d5 02 21
{ r17 = r17
  if (cmp.gtu(r17.new, r21)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 f5 02 21
{ r17 = r17
  if (cmp.gtu(r17.new, r21)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 d5 42 21
{ r17 = r17
  if (!cmp.gtu(r17.new, r21)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 f5 42 21
{ r17 = r17
  if (!cmp.gtu(r17.new, r21)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 d5 82 21
{ r17 = r17
  if (cmp.gt(r21, r17.new)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 f5 82 21
{ r17 = r17
  if (cmp.gt(r21, r17.new)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 d5 c2 21
{ r17 = r17
  if (!cmp.gt(r21, r17.new)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 f5 c2 21
{ r17 = r17
  if (!cmp.gt(r21, r17.new)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 d5 02 22
{ r17 = r17
  if (cmp.gtu(r21, r17.new)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 f5 02 22
{ r17 = r17
  if (cmp.gtu(r21, r17.new)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 d5 42 22
{ r17 = r17
  if (!cmp.gtu(r21, r17.new)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 f5 42 22
{ r17 = r17
  if (!cmp.gtu(r21, r17.new)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 d5 02 24
{ r17 = r17
  if (cmp.eq(r17.new, #21)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 f5 02 24
{ r17 = r17
  if (cmp.eq(r17.new, #21)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 d5 42 24
{ r17 = r17
  if (!cmp.eq(r17.new, #21)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 f5 42 24
{ r17 = r17
  if (!cmp.eq(r17.new, #21)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 d5 82 24
{ r17 = r17
  if (cmp.gt(r17.new, #21)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 f5 82 24
{ r17 = r17
  if (cmp.gt(r17.new, #21)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 d5 c2 24
{ r17 = r17
  if (!cmp.gt(r17.new, #21)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 f5 c2 24
{ r17 = r17
  if (!cmp.gt(r17.new, #21)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 d5 02 25
{ r17 = r17
  if (cmp.gtu(r17.new, #21)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 f5 02 25
{ r17 = r17
  if (cmp.gtu(r17.new, #21)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 d5 42 25
{ r17 = r17
  if (!cmp.gtu(r17.new, #21)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 f5 42 25
{ r17 = r17
  if (!cmp.gtu(r17.new, #21)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 c0 82 25
{ r17 = r17
  if (tstbit(r17.new, #0)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 e0 82 25
{ r17 = r17
  if (tstbit(r17.new, #0)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 c0 c2 25
{ r17 = r17
  if (!tstbit(r17.new, #0)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 e0 c2 25
{ r17 = r17
  if (!tstbit(r17.new, #0)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 c0 02 26
{ r17 = r17
  if (cmp.eq(r17.new, #-1)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 e0 02 26
{ r17 = r17
  if (cmp.eq(r17.new, #-1)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 c0 42 26
{ r17 = r17
  if (!cmp.eq(r17.new, #-1)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 e0 42 26
{ r17 = r17
  if (!cmp.eq(r17.new, #-1)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 c0 82 26
{ r17 = r17
  if (cmp.gt(r17.new, #-1)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 e0 82 26
{ r17 = r17
  if (cmp.gt(r17.new, #-1)) jump:t 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 c0 c2 26
{ r17 = r17
  if (!cmp.gt(r17.new, #-1)) jump:nt 0x0 }
# CHECK: 11 40 71 70
# CHECK-NEXT: 00 e0 c2 26
{ r17 = r17
  if (!cmp.gt(r17.new, #-1)) jump:t 0x0 }
