; RUN: llc < %s -mtriple=arm-- -mcpu=cortex-a8 | FileCheck %s
; XFAIL: *
; PR11364

; vmov s0, r0 + vmov r0, s0 should have been optimized away.
; rdar://9104514

; Peephole leaves a dead vmovsr instruction behind, and depends on linear scan
; to remove it.

define void @t(float %x) nounwind ssp {
entry:
; CHECK-LABEL:     t:
; CHECK-NOT: vmov
; CHECK:     bl
  %0 = bitcast float %x to i32
  %cmp = icmp ult i32 %0, 2139095039
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @doSomething(float %x) nounwind
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare void @doSomething(float)
