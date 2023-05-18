; RUN: llc < %s -mtriple aarch64 | FileCheck %s

; CHECK: a:
; CHECK-NEXT: .zero 1
@a = internal constant <4 x i1> <i1 false, i1 false, i1 false, i1 false>
; CHECK: b:
; CHECK-NEXT: .byte 5
@b = internal constant <4 x i1> <i1 true, i1 false, i1 true, i1 false>
; CHECK: c:
; CHECK-NEXT: .hword 1
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .zero 1
@c = internal constant <24 x i1> <i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>

