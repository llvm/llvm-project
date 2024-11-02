; RUN: opt -mtriple=thumbv8.1m.main-none-none-eabi -hardware-loops %s -o - | FileCheck %s

@g = common local_unnamed_addr global ptr null, align 4

; CHECK-LABEL: counter_too_large
; CHECK-NOT: call void @llvm.set.loop.iterations
; CHECK-NOT: call i32 @llvm.loop.decrement

define i32 @counter_too_large(i64 %n) {
entry:
  %cmp7 = icmp eq i64 %n, 0
  br i1 %cmp7, label %while.end, label %while.body.lr.ph

while.body.lr.ph:
  %0 = load ptr, ptr @g, align 4
  br label %while.body

while.body:
  %i.09 = phi i64 [ 0, %while.body.lr.ph ], [ %inc1, %while.body ]
  %res.08 = phi i32 [ 0, %while.body.lr.ph ], [ %add, %while.body ]
  %idxprom = trunc i64 %i.09 to i32
  %arrayidx = getelementptr inbounds i32, ptr %0, i32 %idxprom
  %1 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %1, %res.08
  %inc1 = add nuw i64 %i.09, 1
  %cmp = icmp ult i64 %inc1, %n
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:
  br label %while.end

while.end:
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %while.end.loopexit ]
  ret i32 %res.0.lcssa
}
