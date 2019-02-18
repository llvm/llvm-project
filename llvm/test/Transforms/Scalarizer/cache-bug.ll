; RUN: opt -scalarizer -S < %s | FileCheck %s
; RUN: opt -passes='function(scalarizer)' -S < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"


; Check that vector element 1 is scalarized correctly from a chain of
; insertelement instructions
define void @func(i32 %x) {
; CHECK-LABEL: @func(
; CHECK-NOT: phi i32 [ %x, %entry ], [ %inc.pos.y, %loop ]
; CHECK:     phi i32 [ %inc, %entry ], [ %inc.pos.y, %loop ]
; CHECK:   ret void
entry:
  %vecinit = insertelement <2 x i32> <i32 0, i32 0>, i32 %x, i32 1
  %inc = add i32 %x, 1
  %0 = insertelement <2 x i32> %vecinit, i32 %inc, i32 1
  br label %loop

loop:
  %pos = phi <2 x i32> [ %0, %entry ], [ %new.pos.y, %loop ]
  %i = phi i32 [ 0, %entry ], [ %new.i, %loop ]
  %pos.y = extractelement <2 x i32> %pos, i32 1
  %inc.pos.y = add i32 %pos.y, 1
  %new.pos.y = insertelement <2 x i32> %pos, i32 %inc.pos.y, i32 1
  %new.i = add i32 %i, 1
  %cmp2 = icmp slt i32 %new.i, 1
  br i1 %cmp2, label %loop, label %exit

exit:
  ret void
}
