; RUN: opt -S -passes=loop-fusion -loop-fusion-peel-max-count=3 < %s | FileCheck %s

; Tests if we are able to fuse two guarded loops which have constant but
; different trip counts. The first two iterations of the first loop should be
; peeled off, and then the loops should be fused together.

@B = common global [1024 x i32] zeroinitializer, align 16

; CHECK-LABEL: void @main(i32* noalias %A)
; CHECK-NEXT:  entry:
; CHECK:         br i1 %cmp4, label %for.first.entry, label %for.end
; CHECK:       for.first.entry
; CHECK-NEXT:    br label %for.first.peel.begin
; CHECK:       for.first.peel.begin:
; CHECK-NEXT:    br label %for.first.peel
; CHECK:       for.first.peel:
; CHECK:         br label %for.first.peel.next
; CHECK:       for.first.peel.next:
; CHECK-NEXT:    br label %for.first.peel2
; CHECK:       for.first.peel2:
; CHECK:         br label %for.first.peel.next1
; CHECK:       for.first.peel.next1:
; CHECK-NEXT:    br label %for.first.peel.next11
; CHECK:       for.first.peel.next11:
; CHECK-NEXT:    br label %for.first.entry.peel.newph
; CHECK:       for.first.entry.peel.newph:
; CHECK:         br label %for.first
; CHECK:       for.first:
; CHECK:         br i1 %cmp3, label %for.first, label %for.second.exit
; CHECK:       for.second.exit:
; CHECK:         br label %for.end
; CHECK:       for.end:
; CHECK-NEXT:    ret void

define void @main(i32* noalias %A) {
entry:
  %cmp4 = icmp slt i64 0, 45
  br i1 %cmp4, label %for.first.entry, label %for.second.guard

for.first.entry:                                 ; preds = %entry
  br label %for.first

for.first:                                       ; preds = %for.first.entry, %for.first
  %i.05 = phi i64 [ %inc, %for.first ], [ 0, %for.first.entry ]
  %sub = sub nsw i64 %i.05, 3
  %add = add nsw i64 %i.05, 3
  %mul = mul nsw i64 %sub, %add
  %rem = srem i64 %mul, %i.05
  %conv = trunc i64 %rem to i32
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.05
  store i32 %conv, i32* %arrayidx, align 4
  %inc = add nsw i64 %i.05, 1
  %cmp = icmp slt i64 %inc, 45
  br i1 %cmp, label %for.first, label %for.first.exit

for.first.exit:                                  ; preds = %for.first
  br label %for.second.guard

for.second.guard:                                ; preds = %for.first.exit, %entry
  %cmp31 = icmp slt i64 2, 45
  br i1 %cmp31, label %for.second.entry, label %for.end

for.second.entry:                                ; preds = %for.second.guard
  br label %for.second

for.second:                                      ; preds = %for.second.entry, %for.second
  %i1.02 = phi i64 [ %inc14, %for.second ], [ 2, %for.second.entry ]
  %sub7 = sub nsw i64 %i1.02, 3
  %add8 = add nsw i64 %i1.02, 3
  %mul9 = mul nsw i64 %sub7, %add8
  %rem10 = srem i64 %mul9, %i1.02
  %conv11 = trunc i64 %rem10 to i32
  %arrayidx12 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %i1.02
  store i32 %conv11, i32* %arrayidx12, align 4
  %inc14 = add nsw i64 %i1.02, 1
  %cmp3 = icmp slt i64 %inc14, 45
  br i1 %cmp3, label %for.second, label %for.second.exit

for.second.exit:                                 ; preds = %for.second
  br label %for.end

for.end:                                         ; preds = %for.second.exit, %for.second.guard
  ret void
}
