; RUN: opt -passes=loop-vectorize -force-vector-width=4 -S -o - < %s | FileCheck %s
%arrayt = type [64 x i32]

@v_146 = external global %arrayt, align 1

; Since the program has well defined behavior, it should not introduce store poison
; CHECK: vector.ph:
; CHECK-NEXT: br label %vector.body
; CHECK: vector.body:
; CHECK: store <4 x i32> zeroinitializer,
; CHECK: br i1 %{{.*}}, label %middle.block, label %vector.body

define void @foo() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %cond.end, %entry
  %storemerge = phi i16 [ 0, %entry ], [ %inc, %cond.end ]
  %cmp = icmp slt i16 %storemerge, 15
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  br i1 true, label %cond.false, label %land.rhs

land.rhs:                                         ; preds = %for.body
  br i1 poison, label %cond.end, label %cond.false

cond.false:                                       ; preds = %for.body, %land.rhs
  br label %cond.end

cond.end:                                         ; preds = %land.rhs, %cond.false
  %cond = phi i32 [ 0, %cond.false ], [ 1, %land.rhs ]
  %arrayidx = getelementptr inbounds %arrayt, ptr @v_146, i16 0, i16 %storemerge
  store i32 %cond, ptr %arrayidx, align 1
  %inc = add nsw i16 %storemerge, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
