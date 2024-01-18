; RUN: opt -S -passes=loop-reroll   %s | FileCheck %s
target triple = "aarch64--linux-gnu"

define void @test(i32 %n, ptr %arrayidx200, ptr %arrayidx164, ptr %arrayidx172) {
entry:
  %rem.i = srem i32 %n, 4
  %t22 = load float, ptr %arrayidx172, align 4
  %cmp.9 = icmp eq i32 %n, 0
  %t7 = sext i32 %n to i64
  br i1 %cmp.9, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
;CHECK-LABEL: while.body:
;CHECK-NEXT:    %indvar = phi i64 [ %indvar.next, %while.body ], [ 0, %while.body.preheader ]
;CHECK-NEXT:    %arrayidx62.i = getelementptr inbounds float, ptr %arrayidx200, i64 %indvar
;CHECK-NEXT:    %t1 = load float, ptr %arrayidx62.i, align 4
;CHECK-NEXT:    %arrayidx64.i = getelementptr inbounds float, ptr %arrayidx164, i64 %indvar
;CHECK-NEXT:    %t2 = load float, ptr %arrayidx64.i, align 4
;CHECK-NEXT:    %mul65.i = fmul fast float %t2, %t22
;CHECK-NEXT:    %add66.i = fadd fast float %mul65.i, %t1
;CHECK-NEXT:    store float %add66.i, ptr %arrayidx62.i, align 4
;CHECK-NEXT:    %indvar.next = add i64 %indvar, 1
;CHECK-NEXT:    %exitcond = icmp eq i64 %indvar, %{{[0-9]+}}
;CHECK-NEXT:    br i1 %exitcond, label %while.end.loopexit, label %while.body

  %indvars.iv.i423 = phi i64 [ %indvars.iv.next.i424, %while.body ], [ 0, %while.body.preheader ]
  %i.22.i = phi i32 [ %add103.i, %while.body ], [ %rem.i, %while.body.preheader ]
  %arrayidx62.i = getelementptr inbounds float, ptr %arrayidx200, i64 %indvars.iv.i423
  %t1 = load float, ptr %arrayidx62.i, align 4
  %arrayidx64.i = getelementptr inbounds float, ptr %arrayidx164, i64 %indvars.iv.i423
  %t2 = load float, ptr %arrayidx64.i, align 4
  %mul65.i = fmul fast float %t2, %t22
  %add66.i = fadd fast float %mul65.i, %t1
  store float %add66.i, ptr %arrayidx62.i, align 4
  %t3 = add nsw i64 %indvars.iv.i423, 1
  %arrayidx71.i = getelementptr inbounds float, ptr %arrayidx200, i64 %t3
  %t4 = load float, ptr %arrayidx71.i, align 4
  %arrayidx74.i = getelementptr inbounds float, ptr %arrayidx164, i64 %t3
  %t5 = load float, ptr %arrayidx74.i, align 4
  %mul75.i = fmul fast float %t5, %t22
  %add76.i = fadd fast float %mul75.i, %t4
  store float %add76.i, ptr %arrayidx71.i, align 4
  %add103.i = add nsw i32 %i.22.i, 2
  %t6 = sext i32 %add103.i to i64
  %cmp58.i = icmp slt i64 %t6, %t7
  %indvars.iv.next.i424 = add i64 %indvars.iv.i423, 2
  br i1 %cmp58.i, label %while.body, label %while.end.loopexit

while.end.loopexit:
  br label %while.end

while.end:
  ret void
}

; Function Attrs: noinline norecurse nounwind
define i32 @test2(i64 %n, ptr nocapture %x, ptr nocapture readonly %y) {
entry:
  %cmp18 = icmp sgt i64 %n, 0
  br i1 %cmp18, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body

;CHECK-LABEL:     for.body:
;CHECK-NEXT:  %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %for.body.preheader ]
;CHECK-NEXT:  %arrayidx = getelementptr inbounds i32, ptr %y, i64 %indvar
;CHECK-NEXT:  [[T1:%[0-9]+]] = load i32, ptr %arrayidx, align 4
;CHECK-NEXT:  %arrayidx3 = getelementptr inbounds i32, ptr %x, i64 %indvar
;CHECK-NEXT:  store i32 [[T1]], ptr %arrayidx3, align 4
;CHECK-NEXT:  %indvar.next = add i64 %indvar, 1
;CHECK-NEXT:  %exitcond = icmp eq i64 %indvar, %{{[0-9]+}}
;CHECK-NEXT:  br i1 %exitcond, label %for.end.loopexit, label %for.body

  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %y, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %x, i64 %indvars.iv
  store i32 %0, ptr %arrayidx3, align 4
  %1 = or disjoint i64 %indvars.iv, 1
  %arrayidx5 = getelementptr inbounds i32, ptr %y, i64 %1
  %2 = load i32, ptr %arrayidx5, align 4
  %arrayidx8 = getelementptr inbounds i32, ptr %x, i64 %1
  store i32 %2, ptr %arrayidx8, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp slt i64 %indvars.iv.next, %n
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret i32 0
}

; Function Attrs: noinline norecurse nounwind
define i32 @test3(i32 %n, ptr nocapture %x, ptr nocapture readonly %y) {
entry:
  %cmp21 = icmp sgt i32 %n, 0
  br i1 %cmp21, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body

;CHECK-LABEL:      for.body:
;CHECK:        %add12 = add i8 %i.022, 2
;CHECK-NEXT:   %conv = sext i8 %add12 to i32
;CHECK-NEXT:   %cmp = icmp slt i32 %conv, %n
;CHECK-NEXT:   br i1 %cmp, label %for.body, label %for.end.loopexit

  %conv23 = phi i32 [ %conv, %for.body ], [ 0, %for.body.preheader ]
  %i.022 = phi i8 [ %add12, %for.body ], [ 0, %for.body.preheader ]
  %idxprom = sext i8 %i.022 to i64
  %arrayidx = getelementptr inbounds i32, ptr %y, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %x, i64 %idxprom
  store i32 %0, ptr %arrayidx3, align 4
  %add = or disjoint i32 %conv23, 1
  %idxprom5 = sext i32 %add to i64
  %arrayidx6 = getelementptr inbounds i32, ptr %y, i64 %idxprom5
  %1 = load i32, ptr %arrayidx6, align 4
  %arrayidx10 = getelementptr inbounds i32, ptr %x, i64 %idxprom5
  store i32 %1, ptr %arrayidx10, align 4
  %add12 = add i8 %i.022, 2
  %conv = sext i8 %add12 to i32
  %cmp = icmp slt i32 %conv, %n
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret i32 0
}

; Function Attrs: noinline norecurse nounwind
define i32 @test4(i64 %n, ptr nocapture %x, ptr nocapture readonly %y) {
entry:
  %cmp18 = icmp eq i64 %n, 0
  br i1 %cmp18, label %for.end, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body

;CHECK-LABEL:     for.body:
;CHECK-NEXT:  %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %for.body.preheader ]
;CHECK-NEXT:  %arrayidx = getelementptr inbounds i32, ptr %y, i64 %indvar
;CHECK-NEXT:  [[T1:%[0-9]+]] = load i32, ptr %arrayidx, align 4
;CHECK-NEXT:  %arrayidx3 = getelementptr inbounds i32, ptr %x, i64 %indvar
;CHECK-NEXT:  store i32 [[T1]], ptr %arrayidx3, align 4
;CHECK-NEXT:  %indvar.next = add i64 %indvar, 1
;CHECK-NEXT:  %exitcond = icmp eq i64 %indvar, %{{[0-9]+}}
;CHECK-NEXT:  br i1 %exitcond, label %for.end.loopexit, label %for.body

  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %y, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %x, i64 %indvars.iv
  store i32 %0, ptr %arrayidx3, align 4
  %1 = or disjoint i64 %indvars.iv, 1
  %arrayidx5 = getelementptr inbounds i32, ptr %y, i64 %1
  %2 = load i32, ptr %arrayidx5, align 4
  %arrayidx8 = getelementptr inbounds i32, ptr %x, i64 %1
  store i32 %2, ptr %arrayidx8, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  %cmp = icmp ult i64 %indvars.iv.next, %n
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret i32 0
}

