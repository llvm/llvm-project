; RUN: llc --verify-machineinstrs %s -o - | FileCheck %s

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-unknown-unknown-eabi"

declare dso_local i32 @h0(i32, i32) local_unnamed_addr #1

define dso_local i32 @f(i32 %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
entry:
  %add = add nsw i32 %a, 1
  %sub = add nsw i32 %b, -1
  %call = tail call i32 @h0(i32 %add, i32 %sub) #0
  %add1 = add nsw i32 %c, %b
  %mul = shl nsw i32 %call, 1
  %add2 = add nsw i32 %mul, %add1
  %sub3 = sub nsw i32 %c, %d
  %mul4 = mul nsw i32 %add2, %sub3
  %sub5 = sub nsw i32 %call, %add1
  %div = sdiv i32 %mul4, %sub5
  %add6 = add nsw i32 %d, %c
  %mul7 = mul nsw i32 %div, %add6
  %add8 = add nsw i32 %mul7, 1
  ret i32 %add8
}
; CHECK-LABEL: f:
; CHECK:       bl   h0
; CHECK-NEXT:  bl   OUTLINED_FUNCTION_0
; CHECK-NEXT:  adds r0, #1
; CHECK-NEXT:  pop  {r4, r5, r6, pc}


define dso_local i32 @g(i32 %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
entry:
  %sub = add nsw i32 %a, -1
  %add = add nsw i32 %b, 1
  %call = tail call i32 @h0(i32 %sub, i32 %add) #0
  %add1 = add nsw i32 %c, %b
  %mul = shl nsw i32 %call, 1
  %add2 = add nsw i32 %mul, %add1
  %sub3 = sub nsw i32 %c, %d
  %mul4 = mul nsw i32 %add2, %sub3
  %sub5 = sub nsw i32 %call, %add1
  %div = sdiv i32 %mul4, %sub5
  %add6 = add nsw i32 %d, %c
  %mul7 = mul nsw i32 %div, %add6
  %add8 = add nsw i32 %mul7, 2
  ret i32 %add8
}
; CHECK-LABEL: g:
; CHECK:       bl   h0
; CHECK-NEXT:  bl   OUTLINED_FUNCTION_0
; CHECK-NEXT:  adds r0, #2
; CHECK-NEXT:  pop  {r4, r5, r6, pc}


attributes #0 = { minsize nounwind optsize }
attributes #1 = { minsize optsize }
