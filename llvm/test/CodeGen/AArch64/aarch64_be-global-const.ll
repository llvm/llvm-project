; RUN: llc -mtriple=aarch64_be < %s | FileCheck %s

; CHECK-LABEL: G:
; CHECK: .byte 11
; CHECK: .size G, 1
@G = global <4 x i1> <i1 true, i1 false, i1 true, i1 true>
