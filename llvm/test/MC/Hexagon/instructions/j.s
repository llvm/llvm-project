# RUN: llvm-mc -triple hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.4 J

# Call subroutine
# CHECK: 00 c0 00 5a
call 0
# CHECK: 00 c3 00 5d
if (p3) call 0
# CHECK: 00 c3 20 5d
if (!p3) call 0

# Compare and jump
# CHECK: 00 c0 89 11
{ p0 = cmp.eq(r17,#-1); if (p0.new) jump:nt 0 }
# CHECK: 00 c1 89 11
{ p0 = cmp.gt(r17,#-1); if (p0.new) jump:nt 0 }
# CHECK: 00 c3 89 11
{ p0 = tstbit(r17, #0); if (p0.new) jump:nt 0 }
# CHECK: 00 e0 89 11
{ p0 = cmp.eq(r17,#-1); if (p0.new) jump:t 0 }
# CHECK: 00 e1 89 11
{ p0 = cmp.gt(r17,#-1); if (p0.new) jump:t 0 }
# CHECK: 00 e3 89 11
{ p0 = tstbit(r17, #0); if (p0.new) jump:t 0 }
# CHECK: 00 c0 c9 11
{ p0 = cmp.eq(r17,#-1); if (!p0.new) jump:nt 0 }
# CHECK: 00 c1 c9 11
{ p0 = cmp.gt(r17,#-1); if (!p0.new) jump:nt 0 }
# CHECK: 00 c3 c9 11
{ p0 = tstbit(r17, #0); if (!p0.new) jump:nt 0 }
# CHECK: 00 e0 c9 11
{ p0 = cmp.eq(r17,#-1); if (!p0.new) jump:t 0 }
# CHECK: 00 e1 c9 11
{ p0 = cmp.gt(r17,#-1); if (!p0.new) jump:t 0 }
# CHECK: 00 e3 c9 11
{ p0 = tstbit(r17, #0); if (!p0.new) jump:t 0 }
# CHECK: 00 d5 09 10
{ p0 = cmp.eq(r17, #21); if (p0.new) jump:nt 0 }
# CHECK: 00 f5 09 10
{ p0 = cmp.eq(r17, #21); if (p0.new) jump:t 0 }
# CHECK: 00 d5 49 10
{ p0 = cmp.eq(r17, #21); if (!p0.new) jump:nt 0 }
# CHECK: 00 f5 49 10
{ p0 = cmp.eq(r17, #21); if (!p0.new) jump:t 0 }
# CHECK: 00 d5 89 10
{ p0 = cmp.gt(r17, #21); if (p0.new) jump:nt 0 }
# CHECK: 00 f5 89 10
{ p0 = cmp.gt(r17, #21); if (p0.new) jump:t 0 }
# CHECK: 00 d5 c9 10
{ p0 = cmp.gt(r17, #21); if (!p0.new) jump:nt 0 }
# CHECK: 00 f5 c9 10
{ p0 = cmp.gt(r17, #21); if (!p0.new) jump:t 0 }
# CHECK: 00 d5 09 11
{ p0 = cmp.gtu(r17, #21); if (p0.new) jump:nt 0 }
# CHECK: 00 f5 09 11
{ p0 = cmp.gtu(r17, #21); if (p0.new) jump:t 0 }
# CHECK: 00 d5 49 11
{ p0 = cmp.gtu(r17, #21); if (!p0.new) jump:nt 0 }
# CHECK: 00 f5 49 11
{ p0 = cmp.gtu(r17, #21); if (!p0.new) jump:t 0 }
# CHECK: 00 c0 89 13
{ p1 = cmp.eq(r17,#-1); if (p1.new) jump:nt 0 }
# CHECK: 00 c1 89 13
{ p1 = cmp.gt(r17,#-1); if (p1.new) jump:nt 0 }
# CHECK: 00 c3 89 13
{ p1 = tstbit(r17, #0); if (p1.new) jump:nt 0 }
# CHECK: 00 e0 89 13
{ p1 = cmp.eq(r17,#-1); if (p1.new) jump:t 0 }
# CHECK: 00 e1 89 13
{ p1 = cmp.gt(r17,#-1); if (p1.new) jump:t 0 }
# CHECK: 00 e3 89 13
{ p1 = tstbit(r17, #0); if (p1.new) jump:t 0 }
# CHECK: 00 c0 c9 13
{ p1 = cmp.eq(r17,#-1); if (!p1.new) jump:nt 0 }
# CHECK: 00 c1 c9 13
{ p1 = cmp.gt(r17,#-1); if (!p1.new) jump:nt 0 }
# CHECK: 00 c3 c9 13
{ p1 = tstbit(r17, #0); if (!p1.new) jump:nt 0 }
# CHECK: 00 e0 c9 13
{ p1 = cmp.eq(r17,#-1); if (!p1.new) jump:t 0 }
# CHECK: 00 e1 c9 13
{ p1 = cmp.gt(r17,#-1); if (!p1.new) jump:t 0 }
# CHECK: 00 e3 c9 13
{ p1 = tstbit(r17, #0); if (!p1.new) jump:t 0 }
# CHECK: 00 d5 09 12
{ p1 = cmp.eq(r17, #21); if (p1.new) jump:nt 0 }
# CHECK: 00 f5 09 12
{ p1 = cmp.eq(r17, #21); if (p1.new) jump:t 0 }
# CHECK: 00 d5 49 12
{ p1 = cmp.eq(r17, #21); if (!p1.new) jump:nt 0 }
# CHECK: 00 f5 49 12
{ p1 = cmp.eq(r17, #21); if (!p1.new) jump:t 0 }
# CHECK: 00 d5 89 12
{ p1 = cmp.gt(r17, #21); if (p1.new) jump:nt 0 }
# CHECK: 00 f5 89 12
{ p1 = cmp.gt(r17, #21); if (p1.new) jump:t 0 }
# CHECK: 00 d5 c9 12
{ p1 = cmp.gt(r17, #21); if (!p1.new) jump:nt 0 }
# CHECK: 00 f5 c9 12
{ p1 = cmp.gt(r17, #21); if (!p1.new) jump:t 0 }
# CHECK: 00 d5 09 13
{ p1 = cmp.gtu(r17, #21); if (p1.new) jump:nt 0 }
# CHECK: 00 f5 09 13
{ p1 = cmp.gtu(r17, #21); if (p1.new) jump:t 0 }
# CHECK: 00 d5 49 13
{ p1 = cmp.gtu(r17, #21); if (!p1.new) jump:nt 0 }
# CHECK: 00 f5 49 13
{ p1 = cmp.gtu(r17, #21); if (!p1.new) jump:t 0 }
# CHECK: 00 cd 09 14
{ p0 = cmp.eq(r17, r21); if (p0.new) jump:nt 0 }
# CHECK: 00 dd 09 14
{ p1 = cmp.eq(r17, r21); if (p1.new) jump:nt 0 }
# CHECK: 00 ed 09 14
{ p0 = cmp.eq(r17, r21); if (p0.new) jump:t 0 }
# CHECK: 00 fd 09 14
{ p1 = cmp.eq(r17, r21); if (p1.new) jump:t 0 }
# CHECK: 00 cd 49 14
{ p0 = cmp.eq(r17, r21); if (!p0.new) jump:nt 0 }
# CHECK: 00 dd 49 14
{ p1 = cmp.eq(r17, r21); if (!p1.new) jump:nt 0 }
# CHECK: 00 ed 49 14
{ p0 = cmp.eq(r17, r21); if (!p0.new) jump:t 0 }
# CHECK: 00 fd 49 14
{ p1 = cmp.eq(r17, r21); if (!p1.new) jump:t 0 }
# CHECK: 00 cd 89 14
{ p0 = cmp.gt(r17, r21); if (p0.new) jump:nt 0 }
# CHECK: 00 dd 89 14
{ p1 = cmp.gt(r17, r21); if (p1.new) jump:nt 0 }
# CHECK: 00 ed 89 14
{ p0 = cmp.gt(r17, r21); if (p0.new) jump:t 0 }
# CHECK: 00 fd 89 14
{ p1 = cmp.gt(r17, r21); if (p1.new) jump:t 0 }
# CHECK: 00 cd c9 14
{ p0 = cmp.gt(r17, r21); if (!p0.new) jump:nt 0 }
# CHECK: 00 dd c9 14
{ p1 = cmp.gt(r17, r21); if (!p1.new) jump:nt 0 }
# CHECK: 00 ed c9 14
{ p0 = cmp.gt(r17, r21); if (!p0.new) jump:t 0 }
# CHECK: 00 fd c9 14
{ p1 = cmp.gt(r17, r21); if (!p1.new) jump:t 0 }
# CHECK: 00 cd 09 15
{ p0 = cmp.gtu(r17, r21); if (p0.new) jump:nt 0 }
# CHECK: 00 dd 09 15
{ p1 = cmp.gtu(r17, r21); if (p1.new) jump:nt 0 }
# CHECK: 00 ed 09 15
{ p0 = cmp.gtu(r17, r21); if (p0.new) jump:t 0 }
# CHECK: 00 fd 09 15
{ p1 = cmp.gtu(r17, r21); if (p1.new) jump:t 0 }
# CHECK: 00 cd 49 15
{ p0 = cmp.gtu(r17, r21); if (!p0.new) jump:nt 0 }
# CHECK: 00 dd 49 15
{ p1 = cmp.gtu(r17, r21); if (!p1.new) jump:nt 0 }
# CHECK: 00 ed 49 15
{ p0 = cmp.gtu(r17, r21); if (!p0.new) jump:t 0 }
# CHECK: 00 fd 49 15
{ p1 = cmp.gtu(r17, r21); if (!p1.new) jump:t 0 }

# Jump to address
# CHECK: 00 c0 00 58
jump 0
# CHECK: 00 c3 00 5c
if (p3) jump 0
# CHECK: 00 c3 20 5c
if (!p3) jump 0

# Jump to address conditioned on new predicate
# CHECK: 03 40 45 85
# CHECK-NEXT: 00 cb 00 5c
{ p3 = r5
  if (p3.new) jump:nt 0 }
# CHECK: 03 40 45 85
# CHECK-NEXT: 00 db 00 5c
{ p3 = r5
  if (p3.new) jump:t 0 }
# CHECK: 03 40 45 85
# CHECK-NEXT: 00 cb 20 5c
{ p3 = r5
  if (!p3.new) jump:nt 0 }
# CHECK: 03 40 45 85
# CHECK-NEXT: 00 db 20 5c
{ p3 = r5
  if (!p3.new) jump:t 0 }

# Jump to address conditioned on register value
# CHECK: 00 c0 11 61
if (r17!=#0) jump:nt 0
# CHECK: 00 d0 11 61
if (r17!=#0) jump:t 0
# CHECK: 00 c0 51 61
if (r17>=#0) jump:nt 0
# CHECK: 00 d0 51 61
if (r17>=#0) jump:t 0
# CHECK: 00 c0 91 61
if (r17==#0) jump:nt 0
# CHECK: 00 d0 91 61
if (r17==#0) jump:t 0
# CHECK: 00 c0 d1 61
if (r17<=#0) jump:nt 0
# CHECK: 00 d0 d1 61
if (r17<=#0) jump:t 0

# Transfer and jump
# CHECK: 00 d5 09 16
{ r17 = #21 ; jump 0 }
# CHECK: 00 c9 0d 17
{ r17 = r21 ; jump 0 }
