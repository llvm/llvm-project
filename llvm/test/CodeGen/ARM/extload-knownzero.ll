; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 %s -o - | FileCheck %s
; rdar://12771555

define void @foo(ptr %ptr, i32 %a) nounwind {
entry:
; CHECK-LABEL: foo:
  %tmp1 = icmp ult i32 %a, 100
  br i1 %tmp1, label %bb1, label %bb2
bb1:
; CHECK: ldrh
  %tmp2 = load i16, ptr %ptr, align 2
  br label %bb2
bb2:
; CHECK-NOT: uxth
; CHECK: cmp
  %tmp3 = phi i16 [ 0, %entry ], [ %tmp2, %bb1 ]
  %cmp = icmp ult i16 %tmp3, 24
  br i1 %cmp, label %bb3, label %exit
bb3:
  call void @bar() nounwind
  br label %exit
exit:
  ret void
}

declare void @bar () 
