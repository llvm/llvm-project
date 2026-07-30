; RUN: opt -passes=licm -S < %s | FileCheck %s

define void @f(i1 zeroext %p1) {
; CHECK-LABEL: @f(
entry:
  br label %lbl

lbl.loopexit:                                     ; No predecessors!
  br label %lbl

lbl:                                              ; preds = %lbl.loopexit, %entry
  %phi = phi i32 [ %conv, %lbl.loopexit ], [ poison, %entry ]
; CHECK: phi i32 [ poison, {{.*}} ], [ poison
  br label %if.then.5

if.then.5:                                        ; preds = %if.then.5, %lbl
  %conv = zext i1 0 to i32
  br label %if.then.5
}
