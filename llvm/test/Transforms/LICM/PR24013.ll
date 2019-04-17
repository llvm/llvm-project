; RUN: opt -licm -S < %s | FileCheck %s

define void @f(i1 zeroext %p1) {
; CHECK-LABEL: @f(
entry:
  br label %lbl

lbl.loopexit:                                     ; No predecessors!
  br label %lbl

lbl:                                              ; preds = %lbl.loopexit, %entry
  %phi = phi i32 [ %conv, %lbl.loopexit ], [ undef, %entry ]
; CHECK: phi i32 [ undef, {{.*}} ], [ undef
  br label %if.then.5

if.then.5:                                        ; preds = %if.then.5, %lbl
  %conv = zext i1 undef to i32
  br label %if.then.5
}
