; RUN: opt -S -passes=loop-fusion -loop-fusion-peel-max-count=3 < %s | FileCheck %s

; Tests that we do not fuse two guarded loops together.
; These loops do not have the same trip count, and the first loop meets the
; requirements for peeling. However, the exit block of the first loop makes the
; loops unsafe to fuse together.
; The expected output of this test is the function as below.

; CHECK-LABEL: void @unsafe_exitblock(ptr noalias %A, ptr noalias %B)
; CHECK:       for.first.guard
; CHECK:         br i1 %cmp3, label %for.first.preheader, label %for.second.guard
; CHECK:       for.first.preheader:
; CHECK-NEXT:    br label %for.first
; CHECK:       for.first:
; CHECK:         br i1 %cmp, label %for.first, label %for.first.exit
; CHECK:       for.first.exit:
; CHECK-NEXT:    call void @bar()
; CHECK-NEXT:    br label %for.second.guard
; CHECK:       for.second.guard:
; CHECK:         br i1 %cmp21, label %for.second.preheader, label %for.end
; CHECK:       for.second.preheader:
; CHECK-NEXT:    br label %for.second
; CHECK:       for.second:
; CHECK:         br i1 %cmp2, label %for.second, label %for.second.exit
; CHECK:       for.second.exit:
; CHECK-NEXT:    br label %for.end
; CHECK:       for.end:
; CHECK-NEXT:    ret void

define void @unsafe_exitblock(ptr noalias %A, ptr noalias %B) {
for.first.guard:
  %cmp3 = icmp slt i64 0, 45
  br i1 %cmp3, label %for.first.preheader, label %for.second.guard

for.first.preheader:                             ; preds = %for.first.guard
  br label %for.first

for.first:                                       ; preds = %for.first.preheader, %for.first
  %i.04 = phi i64 [ %inc, %for.first ], [ 0, %for.first.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %i.04
  store i32 0, ptr %arrayidx, align 4
  %inc = add nsw i64 %i.04, 1
  %cmp = icmp slt i64 %inc, 45
  br i1 %cmp, label %for.first, label %for.first.exit

for.first.exit:                                  ; preds = %for.first
  call void @bar()
  br label %for.second.guard

for.second.guard:                                ; preds = %for.first.exit, %for.first.guard
  %cmp21 = icmp slt i64 2,45
  br i1 %cmp21, label %for.second.preheader, label %for.end

for.second.preheader:                            ; preds = %for.second.guard
  br label %for.second

for.second:                                      ; preds = %for.second.preheader, %for.second
  %j.02 = phi i64 [ %inc6, %for.second ], [ 2, %for.second.preheader ]
  %arrayidx4 = getelementptr inbounds i32, ptr %B, i64 %j.02
  store i32 0, ptr %arrayidx4, align 4
  %inc6 = add nsw i64 %j.02, 1
  %cmp2 = icmp slt i64 %inc6, 45
  br i1 %cmp2, label %for.second, label %for.second.exit

for.second.exit:                                 ; preds = %for.second
  br label %for.end

for.end:                                         ; preds = %for.second.exit, %for.second.guard
  ret void
}

declare void @bar()
