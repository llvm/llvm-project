; RUN: opt < %s -loop-interchange -simplifycfg -S | FileCheck %s

; no_deps_interchange just access a single nested array and can be interchange.
define i32 @no_deps_interchange([1024 x i32]* nocapture %Arr, i32 %k) local_unnamed_addr #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.cond.cleanup3
  %indvars.iv19 = phi i64 [ 0, %entry ], [ %indvars.iv.next20, %for.cond.cleanup3 ]
  br label %for.body4

for.body4:                                        ; preds = %for.body, %for.body4
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body4 ]
  %arrayidx6 = getelementptr inbounds [1024 x i32], [1024 x i32]* %Arr, i64 %indvars.iv, i64 %indvars.iv19
  store i32 0, i32* %arrayidx6, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.body4, label %for.cond.cleanup3

for.cond.cleanup3:                                ; preds = %for.body4
  %indvars.iv.next20 = add nuw nsw i64 %indvars.iv19, 1
  %exitcond21 = icmp ne i64 %indvars.iv.next20, 1024
  br i1 %exitcond21, label %for.body, label %for.cond.cleanup


for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret i32 0
}

; CHECK-LABEL: @no_deps_interchange
; CHECK-LABEL: entry:
; CHECK-NEXT: br label %for.body4

; CHECK-LABEL: for.body:                                         ; preds = %for.body4, %for.body
; CHECK: %indvars.iv19 = phi i64 [ %indvars.iv.next20, %for.body ], [ 0, %for.body4 ]
; CHECK: br i1 %exitcond21, label %for.body, label %for.body4.split

; CHECK-LABEL: for.body4:                                        ; preds = %entry, %for.body4.split
; CHECK: %indvars.iv = phi i64 [ %indvars.iv.next, %for.body4.split ], [ 0, %entry ]
; CHECK: br label %for.body

; CHECK-LABEL: for.body4.split:                                  ; preds = %for.body
; CHECK: %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK: br i1 %exitcond, label %for.body4, label %for.cond.cleanup
