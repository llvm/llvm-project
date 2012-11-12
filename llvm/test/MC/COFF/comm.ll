; RUN: llc -mtriple i386-pc-mingw32 < %s | FileCheck %s

@a = internal global i8 0, align 1
@b = internal global double 0.000000e+00, align 8
@c = common global i8 0, align 1
@d = common global double 0.000000e+00, align 8

; .lcomm uses byte alignment
; CHECK: .lcomm	_a,1
; CHECK: .lcomm	_b,8,8
; .comm uses log2 alignment
; CHECK: .comm	_c,1,0
; CHECK: .comm	_d,8,3
