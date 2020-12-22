; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s
; PR1796

declare void @Finisher(i32) noreturn

; Make sure we optimize a sequence of two calls (second unreachable);
define void @double_call(i32) {
; CHECK-LABEL: @double_call(
; CHECK-NEXT:    tail call void @Finisher(i32 %0) #0
; CHECK-NEXT:    unreachable
;
  tail call void @Finisher(i32 %0) noreturn
  tail call void @Finisher(i32 %0) noreturn
  ret void
}

; Make sure we DON'T try to optimize a musttail call (the IR invariant
; is that it must be followed by [optional bitcast then] ret).
define void @must_tail(i32) {
; CHECK-LABEL: @must_tail(
; CHECK-NEXT:    musttail call void @Finisher(i32 %0) #0
; CHECK-NEXT:    ret void
;
  musttail call void @Finisher(i32 %0) #0
  ret void
}

; CHECK: attributes #0 = { noreturn }
attributes #0 = { noreturn }
