; RUN: llc -verify-machineinstrs %s -o - | FileCheck %s
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-unknown-unknown-eabi"

declare dso_local i32 @t(i32) local_unnamed_addr #0

define dso_local i32 @f(i32 %a, i32 %b, i32 %c) local_unnamed_addr #0 {
entry:
  %mul = mul nsw i32 %a, 3
  %add = add nsw i32 %mul, 1
  %sub = add nsw i32 %b, -1
  %div = sdiv i32 %add, %sub
  %sub1 = sub nsw i32 %a, %c
  %div2 = sdiv i32 %div, %sub1
  %mul3 = mul nsw i32 %div2, %b
  %add4 = add nsw i32 %mul3, 1
  %call = tail call i32 @t(i32 %add4) #0
  ret i32 %call
}
; CHECK-LABEL: f:
; CHECK:       str  lr, [sp, #-8]!
; CHECK-NEXT:  bl   OUTLINED_FUNCTION_0
; CHECK-NEXT:  ldr  lr, [sp], #8
; CHECK-NEXT:  adds r0, #1
; CHECK-NEXT:  b    t

define dso_local i32 @g(i32 %a, i32 %b, i32 %c) local_unnamed_addr #0 {
entry:
  %mul = mul nsw i32 %a, 3
  %add = add nsw i32 %mul, 1
  %sub = add nsw i32 %b, -1
  %div = sdiv i32 %add, %sub
  %sub1 = sub nsw i32 %a, %c
  %div2 = sdiv i32 %div, %sub1
  %mul3 = mul nsw i32 %div2, %b
  %add4 = add nsw i32 %mul3, 3
  %call = tail call i32 @t(i32 %add4) #0
  ret i32 %call
}

; CHECK-LABEL: g:
; CHECK:       str  lr, [sp, #-8]!
; CHECK-NEXT:  bl   OUTLINED_FUNCTION_0
; CHECK-NEXT:  ldr  lr, [sp], #8
; CHECK-NEXT:  adds r0, #3
; CHECK-NEXT:  b    t

; CHECK-LABEL: OUTLINED_FUNCTION_0:
; CHECK-NOT:   lr
; CHECK:       bx lr

attributes #0 = { minsize nounwind optsize }
