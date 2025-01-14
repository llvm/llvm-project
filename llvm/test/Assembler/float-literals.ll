; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: @a = global float -0.000000e+00
@a = global float -0.0
; CHECK: @b = global float 0.000000e+00
@b = global float +0.0
; CHECK: @c = global float 0.000000e+00
@c = global float 0.0
; CHECK: @d = global float 0.000000e+00
@d = global float 0.e1
; CHECK: @e = global float 0.000000e+00
@e = global float 0.e-1
; CHECK: @f = global float 0.000000e+00
@f = global float 0.e+1
; CHECK: @g = global float 0x3DF0000000000000
@g = global float 0x1.0p-32
; CHECK: @h = global float 0x41F0000000000000
@h = global float 0x1.0p+32
; CHECK: @i = global float 0x41FC300000000000
@i = global float 0x1.c3p32
; CHECK: @j = global float 0x3FFFF00000000000
@j = global float 0x1.ffp0
; CHECK: @k = global float 0xC0FFFFFFE0000000
@k = global float -0xfff.fffp5
; CHECK: @l = global float 0x4080FDE000000000
@l = global float +0x10.fdep5

; CHECK: @0 = global double 0x7FF0000000000000
@0 = global double +inf
; CHECK: @1 = global ppc_fp128 0xMFFF00000000000000000000000000000
@1 = global ppc_fp128 -inf
; CHECK: @2 = global half 0xHFE00
@2 = global half -qnan
; CHECK: @3 = global bfloat 0xR7FC0
@3 = global bfloat +qnan
; CHECK: @4 = global fp128 0xL00000000DEADBEEF7FFF800000000000
@4 = global fp128 +nan(0xdeadbeef)
; CHECK: @5 = global float 0x7FF000002000000
@5 = global float +snan(0x1)
; CHECK: @6 = global x86_fp80 0xK0001FFFF000000000000
@6 = global x86_fp80 f0x0000ffff000000000000

