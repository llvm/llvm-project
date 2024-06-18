; RUN: llc < %s | FileCheck %s

; ModuleID = 'aarch64_tree_tests.bc'
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "arm64--linux-gnu"

; CHECK-LABEL: .LCPI0_0:
; CHECK-NEXT:    .hword 32768
; CHECK-NEXT:    .hword 32767
; CHECK-NEXT:    .hword 4664
; CHECK-NEXT:    .hword 32767
; CHECK-NEXT:    .hword 32768
; CHECK-NEXT:    .hword 32768
; CHECK-NEXT:    .hword 0
; CHECK-NEXT:    .hword 0

; Function Attrs: nounwind readnone
define <8 x i16> @aarch64_tree_tests_and(<8 x i16> %a) {
entry:
  %and = and <8 x i16> <i16 0, i16 undef, i16 undef, i16 0, i16 0, i16 undef, i16 undef, i16 0>, %a
  %ret = add <8 x i16> %and, <i16 -32768, i16 32767, i16 4664, i16 32767, i16 -32768, i16 -32768, i16 0, i16 0>
  ret <8 x i16> %ret
}

; CHECK-LABEL: .LCPI1_0:
; CHECK-NEXT:    .hword 32768
; CHECK-NEXT:    .hword 32766
; CHECK-NEXT:    .hword 4664
; CHECK-NEXT:    .hword 32766
; CHECK-NEXT:    .hword 32768
; CHECK-NEXT:    .hword 32768
; CHECK-NEXT:    .hword 65535
; CHECK-NEXT:    .hword 65535

; Function Attrs: nounwind readnone
define <8 x i16> @aarch64_tree_tests_or(<8 x i16> %a) {
entry:
  %or = or <8 x i16> <i16 -1, i16 undef, i16 undef, i16 -1, i16 -1, i16 undef, i16 undef, i16 -1>, %a
  %ret = add <8 x i16> %or, <i16 -32767, i16 32767, i16 4665, i16 32767, i16 -32767, i16 -32767, i16 0, i16 0>
  ret <8 x i16> %ret
}

