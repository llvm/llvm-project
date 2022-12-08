; RUN: opt -S -passes=loop-fusion < %s | FileCheck %s

@B = common global [1024 x i32] zeroinitializer, align 16

; CHECK: void @dep_free_parametric
; CHECK-next: entry:
; CHECK: br i1 %{{.*}}, label %[[LOOP1PREHEADER:bb[0-9]*]], label %[[LOOP1SUCC:bb[0-9]+]]
; CHECK: [[LOOP1PREHEADER]]
; CHECK-NEXT: br label %[[LOOP1BODY:bb[0-9]*]]
; CHECK: [[LOOP1BODY]]
; CHECK: br i1 %{{.*}}, label %[[LOOP1BODY]], label %[[LOOP2EXIT:bb[0-9]+]]
; CHECK: [[LOOP2EXIT]]
; CHECK: br label %[[LOOP1SUCC]]
; CHECK: [[LOOP1SUCC]]
; CHECK: ret void
define void @dep_free_parametric(i32* noalias %A, i64 %N) {
entry:
  %cmp4 = icmp slt i64 0, %N
  br i1 %cmp4, label %bb3, label %bb14

bb3:                               ; preds = %entry
  br label %bb5

bb5:                                         ; preds = %bb3, %bb5
  %i.05 = phi i64 [ %inc, %bb5 ], [ 0, %bb3 ]
  %sub = sub nsw i64 %i.05, 3
  %add = add nsw i64 %i.05, 3
  %mul = mul nsw i64 %sub, %add
  %rem = srem i64 %mul, %i.05
  %conv = trunc i64 %rem to i32
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.05
  store i32 %conv, i32* %arrayidx, align 4
  %inc = add nsw i64 %i.05, 1
  %cmp = icmp slt i64 %inc, %N
  br i1 %cmp, label %bb5, label %bb10

bb10:                                 ; preds = %bb5
  br label %bb14

bb14:                                          ; preds = %bb10, %entry
  %cmp31 = icmp slt i64 0, %N
  br i1 %cmp31, label %bb8, label %bb12

bb8:                              ; preds = %bb14
  br label %bb9

bb9:                                        ; preds = %bb8, %bb9
  %i1.02 = phi i64 [ %inc14, %bb9 ], [ 0, %bb8 ]
  %sub7 = sub nsw i64 %i1.02, 3
  %add8 = add nsw i64 %i1.02, 3
  %mul9 = mul nsw i64 %sub7, %add8
  %rem10 = srem i64 %mul9, %i1.02
  %conv11 = trunc i64 %rem10 to i32
  %arrayidx12 = getelementptr inbounds [1024 x i32], [1024 x i32]* @B, i64 0, i64 %i1.02
  store i32 %conv11, i32* %arrayidx12, align 4
  %inc14 = add nsw i64 %i1.02, 1
  %cmp3 = icmp slt i64 %inc14, %N
  br i1 %cmp3, label %bb9, label %bb15

bb15:                               ; preds = %bb9
  br label %bb12

bb12:                                        ; preds = %bb15, %bb14
  ret void
}

; Test that `%add` is moved in for.first.preheader, and the two loops for.first
; and for.second are fused.

; CHECK: void @moveinsts_preheader
; CHECK-LABEL: for.first.guard:
; CHECK: br i1 %cmp.guard, label %for.first.preheader, label %for.end
; CHECK-LABEL: for.first.preheader:
; CHECK-NEXT:  %add = add nsw i32 %x, 1
; CHECK-NEXT:  br label %for.first
; CHECK-LABEL: for.first:
; CHECK:   br i1 %cmp.j, label %for.first, label %for.second.exit
; CHECK-LABEL: for.second.exit:
; CHECK-NEXT:   br label %for.end
; CHECK-LABEL: for.end:
; CHECK-NEXT:   ret void
define void @moveinsts_preheader(i32* noalias %A, i32* noalias %B, i64 %N, i32 %x) {
for.first.guard:
  %cmp.guard = icmp slt i64 0, %N
  br i1 %cmp.guard, label %for.first.preheader, label %for.second.guard

for.first.preheader:
  br label %for.first

for.first:
  %i = phi i64 [ %inc.i, %for.first ], [ 0, %for.first.preheader ]
  %Ai = getelementptr inbounds i32, i32* %A, i64 %i
  store i32 0, i32* %Ai, align 4
  %inc.i = add nsw i64 %i, 1
  %cmp.i = icmp slt i64 %inc.i, %N
  br i1 %cmp.i, label %for.first, label %for.first.exit

for.first.exit:
  br label %for.second.guard

for.second.guard:
  br i1 %cmp.guard, label %for.second.preheader, label %for.end

for.second.preheader:
  %add = add nsw i32 %x, 1
  br label %for.second

for.second:
  %j = phi i64 [ %inc.j, %for.second ], [ 0, %for.second.preheader ]
  %Bj = getelementptr inbounds i32, i32* %B, i64 %j
  store i32 0, i32* %Bj, align 4
  %inc.j = add nsw i64 %j, 1
  %cmp.j = icmp slt i64 %inc.j, %N
  br i1 %cmp.j, label %for.second, label %for.second.exit

for.second.exit:
  br label %for.end

for.end:
  ret void
}

; Test that `%add` is moved in for.second.exit, and the two loops for.first
; and for.second are fused.

; CHECK: void @moveinsts_exitblock
; CHECK-LABEL: for.first.guard:
; CHECK: br i1 %cmp.guard, label %for.first.preheader, label %for.end
; CHECK-LABEL: for.first.preheader:
; CHECK-NEXT:  br label %for.first
; CHECK-LABEL: for.first:
; CHECK:   br i1 %cmp.j, label %for.first, label %for.second.exit
; CHECK-LABEL: for.second.exit:
; CHECK-NEXT:  %add = add nsw i32 %x, 1
; CHECK-NEXT:   br label %for.end
; CHECK-LABEL: for.end:
; CHECK-NEXT:   ret void
define void @moveinsts_exitblock(i32* noalias %A, i32* noalias %B, i64 %N, i32 %x) {
for.first.guard:
  %cmp.guard = icmp slt i64 0, %N
  br i1 %cmp.guard, label %for.first.preheader, label %for.second.guard

for.first.preheader:
  br label %for.first

for.first:
  %i.04 = phi i64 [ %inc, %for.first ], [ 0, %for.first.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.04
  store i32 0, i32* %arrayidx, align 4
  %inc = add nsw i64 %i.04, 1
  %cmp = icmp slt i64 %inc, %N
  br i1 %cmp, label %for.first, label %for.first.exit

for.first.exit:
  %add = add nsw i32 %x, 1
  br label %for.second.guard

for.second.guard:
  br i1 %cmp.guard, label %for.second.preheader, label %for.end

for.second.preheader:
  br label %for.second

for.second:
  %j.02 = phi i64 [ %inc6, %for.second ], [ 0, %for.second.preheader ]
  %arrayidx4 = getelementptr inbounds i32, i32* %B, i64 %j.02
  store i32 0, i32* %arrayidx4, align 4
  %inc6 = add nsw i64 %j.02, 1
  %cmp.j = icmp slt i64 %inc6, %N
  br i1 %cmp.j, label %for.second, label %for.second.exit

for.second.exit:
  br label %for.end

for.end:
  ret void
}

; Test that `%add` is moved in for.first.guard, and the two loops for.first
; and for.second are fused.

; CHECK: void @moveinsts_guardblock
; CHECK-LABEL: for.first.guard:
; CHECK-NEXT: %cmp.guard = icmp slt i64 0, %N
; CHECK-NEXT:  %add = add nsw i32 %x, 1
; CHECK: br i1 %cmp.guard, label %for.first.preheader, label %for.end
; CHECK-LABEL: for.first.preheader:
; CHECK-NEXT:  br label %for.first
; CHECK-LABEL: for.first:
; CHECK:   br i1 %cmp.j, label %for.first, label %for.second.exit
; CHECK-LABEL: for.second.exit:
; CHECK-NEXT:   br label %for.end
; CHECK-LABEL: for.end:
; CHECK-NEXT:   ret void
define void @moveinsts_guardblock(i32* noalias %A, i32* noalias %B, i64 %N, i32 %x) {
for.first.guard:
  %cmp.guard = icmp slt i64 0, %N
  br i1 %cmp.guard, label %for.first.preheader, label %for.second.guard

for.first.preheader:
  br label %for.first

for.first:
  %i.04 = phi i64 [ %inc, %for.first ], [ 0, %for.first.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.04
  store i32 0, i32* %arrayidx, align 4
  %inc = add nsw i64 %i.04, 1
  %cmp = icmp slt i64 %inc, %N
  br i1 %cmp, label %for.first, label %for.first.exit

for.first.exit:
  br label %for.second.guard

for.second.guard:
  %add = add nsw i32 %x, 1
  br i1 %cmp.guard, label %for.second.preheader, label %for.end

for.second.preheader:
  br label %for.second

for.second:
  %j.02 = phi i64 [ %inc6, %for.second ], [ 0, %for.second.preheader ]
  %arrayidx4 = getelementptr inbounds i32, i32* %B, i64 %j.02
  store i32 0, i32* %arrayidx4, align 4
  %inc6 = add nsw i64 %j.02, 1
  %cmp.j = icmp slt i64 %inc6, %N
  br i1 %cmp.j, label %for.second, label %for.second.exit

for.second.exit:
  br label %for.end

for.end:
  ret void
}

; Test that the incoming block of `%j.lcssa` is updated correctly
; from for.second.guard to for.first.guard, and the two loops for.first and
; for.second are fused.

; CHECK: i64 @updatephi_guardnonloopblock
; CHECK-LABEL: for.first.guard:
; CHECK-NEXT: %cmp.guard = icmp slt i64 0, %N
; CHECK: br i1 %cmp.guard, label %for.first.preheader, label %for.end
; CHECK-LABEL: for.first.preheader:
; CHECK-NEXT:  br label %for.first
; CHECK-LABEL: for.first:
; CHECK:   br i1 %cmp.j, label %for.first, label %for.second.exit
; CHECK-LABEL: for.second.exit:
; CHECK-NEXT:   br label %for.end
; CHECK-LABEL: for.end:
; CHECK-NEXT:   %j.lcssa = phi i64 [ 0, %for.first.guard ], [ %j.02, %for.second.exit ]
; CHECK-NEXT:   ret i64 %j.lcssa

define i64 @updatephi_guardnonloopblock(i32* noalias %A, i32* noalias %B, i64 %N, i32 %x) {
for.first.guard:
  %cmp.guard = icmp slt i64 0, %N
  br i1 %cmp.guard, label %for.first.preheader, label %for.second.guard

for.first.preheader:
  br label %for.first

for.first:
  %i.04 = phi i64 [ %inc, %for.first ], [ 0, %for.first.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %i.04
  store i32 0, i32* %arrayidx, align 4
  %inc = add nsw i64 %i.04, 1
  %cmp = icmp slt i64 %inc, %N
  br i1 %cmp, label %for.first, label %for.first.exit

for.first.exit:
  br label %for.second.guard

for.second.guard:
  br i1 %cmp.guard, label %for.second.preheader, label %for.end

for.second.preheader:
  br label %for.second

for.second:
  %j.02 = phi i64 [ %inc6, %for.second ], [ 0, %for.second.preheader ]
  %arrayidx4 = getelementptr inbounds i32, i32* %B, i64 %j.02
  store i32 0, i32* %arrayidx4, align 4
  %inc6 = add nsw i64 %j.02, 1
  %cmp.j = icmp slt i64 %inc6, %N
  br i1 %cmp.j, label %for.second, label %for.second.exit

for.second.exit:
  br label %for.end

for.end:
  %j.lcssa = phi i64 [ 0, %for.second.guard ], [ %j.02, %for.second.exit ]
  ret i64 %j.lcssa
}
