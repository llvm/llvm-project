; RUN: opt -hardware-loops -force-hardware-loops=true -hardware-loop-decrement=1 -hardware-loop-counter-bitwidth=32 -S %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-DEC
; RUN: opt -hardware-loops -force-hardware-loops=true -hardware-loop-decrement=1 -hardware-loop-counter-bitwidth=32 -force-hardware-loop-phi=true -S %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-REGDEC
; RUN: opt -hardware-loops -force-hardware-loops=true -hardware-loop-decrement=1 -hardware-loop-counter-bitwidth=32 -force-nested-hardware-loop=true -S %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-DEC --check-prefix=CHECK-NESTED
; RUN: opt -hardware-loops -force-hardware-loops=true -hardware-loop-decrement=1 -hardware-loop-counter-bitwidth=32 -force-hardware-loop-guard=true -S %s -o - | FileCheck %s --check-prefix=CHECK-GUARD
; RUN: opt -hardware-loops -force-hardware-loops=true -hardware-loop-decrement=1 -hardware-loop-counter-bitwidth=32 -force-hardware-loop-phi=true -force-hardware-loop-guard=true -S %s -o - | FileCheck %s --check-prefix=CHECK-GUARD

; CHECK-LABEL: while_lt
define void @while_lt(i32 %i, i32 %N, i32* nocapture %A) {
entry:
  %cmp4 = icmp ult i32 %i, %N
  br i1 %cmp4, label %while.body, label %while.end

; CHECK-GUARD-LABEL: while_lt
; CHECK-GUARD: [[COUNT:%[^ ]+]] = sub i32 %N, %i
; CHECK-GUARD: call void @llvm.set.loop.iterations.i32(i32 [[COUNT]])
; CHECK-GUARD: br label %while.body

; CHECK: while.body.preheader:
; CHECK: [[COUNT:%[^ ]+]] = sub i32 %N, %i
; CHECK: call void @llvm.set.loop.iterations.i32(i32 [[COUNT]])
; CHECK: br label %while.body

; CHECK-REGDEC: [[REM:%[^ ]+]] = phi i32 [ [[COUNT]], %while.body.preheader ], [ [[LOOP_DEC:%[^ ]+]], %while.body ]
; CHECK-REGDEC: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[REM]], i32 1)
; CHECK-REGDEC: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK-REGDEC: br i1 [[CMP]], label %while.body, label %while.end

; CHECK-DEC: [[LOOP_DEC:%[^ ]+]] = call i1 @llvm.loop.decrement.i32(i32 1)
; CHECK-DEC: br i1 [[LOOP_DEC]], label %while.body, label %while.end

while.body:
  %i.addr.05 = phi i32 [ %inc, %while.body ], [ %i, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.addr.05
  store i32 %i.addr.05, i32* %arrayidx, align 4
  %inc = add nuw i32 %i.addr.05, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %while.end, label %while.body

while.end:
  ret void
}

; CHECK-LABEL: while_gt
; CHECK: while.body.preheader:
; CHECK: [[COUNT:%[^ ]+]] = sub i32 %i, %N
; CHECK: call void @llvm.set.loop.iterations.i32(i32 [[COUNT]])
; CHECK: br label %while.body

; CHECK-REGDEC: [[REM:%[^ ]+]] = phi i32 [ [[COUNT]], %while.body.preheader ], [ [[LOOP_DEC:%[^ ]+]], %while.body ]
; CHECK-REGDEC: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[REM]], i32 1)
; CHECK-REGDEC: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK-REGDEC: br i1 [[CMP]], label %while.body, label %while.end

; CHECK-DEC: [[LOOP_DEC:%[^ ]+]] = call i1 @llvm.loop.decrement.i32(i32 1)
; CHECK-DEC: br i1 [[LOOP_DEC]], label %while.body, label %while.end

define void @while_gt(i32 %i, i32 %N, i32* nocapture %A) {
entry:
  %cmp4 = icmp sgt i32 %i, %N
  br i1 %cmp4, label %while.body, label %while.end

while.body:
  %i.addr.05 = phi i32 [ %dec, %while.body ], [ %i, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.addr.05
  store i32 %i.addr.05, i32* %arrayidx, align 4
  %dec = add nsw i32 %i.addr.05, -1
  %cmp = icmp sgt i32 %dec, %N
  br i1 %cmp, label %while.body, label %while.end

while.end:
  ret void
}

; CHECK-GUARD-LABEL: while_gte
; CHECK-GUARD: entry:
; CHECK-GUARD:   br i1 %cmp4, label %while.end, label %while.body.preheader
; CHECK-GUARD: while.body.preheader:
; CHECK-GUARD:   [[ADD:%[^ ]+]] = add i32 %i, 1
; CHECK-GUARD:   [[SEL:%[^ ]+]] = icmp slt i32 %N, %i
; CHECK-GUARD:   [[MIN:%[^ ]+]] = select i1 [[SEL]], i32 %N, i32 %i
; CHECK-GUARD:   [[COUNT:%[^ ]+]] = sub i32 [[ADD]], [[MIN]]
; CHECK-GUARD:   call void @llvm.set.loop.iterations.i32(i32 [[COUNT]])
; CHECK-GUARD:   br label %while.body

; CHECK-LABEL: while_gte
; CHECK: while.body.preheader:
; CHECK: [[ADD:%[^ ]+]] = add i32 %i, 1
; CHECK: [[SEL:%[^ ]+]] = icmp slt i32 %N, %i
; CHECK: [[MIN:%[^ ]+]] = select i1 [[SEL]], i32 %N, i32 %i
; CHECK: [[COUNT:%[^ ]+]] = sub i32 [[ADD]], [[MIN]]
; CHECK: call void @llvm.set.loop.iterations.i32(i32 [[COUNT]])
; CHECK: br label %while.body

; CHECK-REGDEC: [[REM:%[^ ]+]] = phi i32 [ [[COUNT]], %while.body.preheader ], [ [[LOOP_DEC:%[^ ]+]], %while.body ]
; CHECK-REGDEC: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[REM]], i32 1)
; CHECK-REGDEC: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK-REGDEC: br i1 [[CMP]], label %while.body, label %while.end

; CHECK-DEC: [[LOOP_DEC:%[^ ]+]] = call i1 @llvm.loop.decrement.i32(i32 1)
; CHECK-DEC: br i1 [[LOOP_DEC]], label %while.body, label %while.end

define void @while_gte(i32 %i, i32 %N, i32* nocapture %A) {
entry:
  %cmp4 = icmp slt i32 %i, %N
  br i1 %cmp4, label %while.end, label %while.body

while.body:
  %i.addr.05 = phi i32 [ %dec, %while.body ], [ %i, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.addr.05
  store i32 %i.addr.05, i32* %arrayidx, align 4
  %dec = add nsw i32 %i.addr.05, -1
  %cmp = icmp sgt i32 %i.addr.05, %N
  br i1 %cmp, label %while.body, label %while.end

while.end:
  ret void
}

; CHECK-GUARD-LABEL: while_ne
; CHECK-GUARD: entry:
; CHECK-GUARD:   [[TEST:%[^ ]+]] = call i1 @llvm.test.set.loop.iterations.i32(i32 %N)
; CHECK-GUARD:   br i1 [[TEST]], label %while.body.preheader, label %while.end
; CHECK-GUARD: while.body.preheader:
; CHECK-GUARD:   br label %while.body
define void @while_ne(i32 %N, i32* nocapture %A) {
entry:
  %cmp = icmp ne i32 %N, 0
  br i1 %cmp, label %while.body, label %while.end

while.body:
  %i.addr.05 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.addr.05
  store i32 %i.addr.05, i32* %arrayidx, align 4
  %inc = add nuw i32 %i.addr.05, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %while.end, label %while.body

while.end:
  ret void
}

; CHECK-GUARD-LABEL: while_eq
; CHECK-GUARD: entry:
; CHECK-GUARD:   [[TEST:%[^ ]+]] = call i1 @llvm.test.set.loop.iterations.i32(i32 %N)
; CHECK-GUARD:   br i1 [[TEST]], label %while.body.preheader, label %while.end
; CHECK-GUARD: while.body.preheader:
; CHECK-GUARD:   br label %while.body
define void @while_eq(i32 %N, i32* nocapture %A) {
entry:
  %cmp = icmp eq i32 %N, 0
  br i1 %cmp, label %while.end, label %while.body

while.body:
  %i.addr.05 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.addr.05
  store i32 %i.addr.05, i32* %arrayidx, align 4
  %inc = add nuw i32 %i.addr.05, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %while.end, label %while.body

while.end:
  ret void
}

; CHECK-GUARD-LABEL: while_preheader_eq
; CHECK-GUARD: entry:
; CHECK-GUARD:   br label %preheader
; CHECK-GUARD: preheader:
; CHECK-GUARD:   [[TEST:%[^ ]+]] = call i1 @llvm.test.set.loop.iterations.i32(i32 %N)
; CHECK-GUARD:   br i1 [[TEST]], label %while.body.preheader, label %while.end
; CHECK-GUARD: while.body.preheader:
; CHECK-GUARD:   br label %while.body
define void @while_preheader_eq(i32 %N, i32* nocapture %A) {
entry:
  br label %preheader

preheader:
  %cmp = icmp eq i32 %N, 0
  br i1 %cmp, label %while.end, label %while.body

while.body:
  %i.addr.05 = phi i32 [ %inc, %while.body ], [ 0, %preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.addr.05
  store i32 %i.addr.05, i32* %arrayidx, align 4
  %inc = add nuw i32 %i.addr.05, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %while.end, label %while.body

while.end:
  ret void
}

; CHECK-LABEL: nested
; CHECK-NESTED: call void @llvm.set.loop.iterations.i32(i32 %N)
; CHECK-NESTED: br label %while.cond1.preheader.us

; CHECK: call void @llvm.set.loop.iterations.i32(i32 %N)
; CHECK: br label %while.body3.us

; CHECK-DEC: [[LOOP_DEC:%[^ ]+]] = call i1 @llvm.loop.decrement.i32(i32 1)

; CHECK-REGDEC: [[REM:%[^ ]+]] = phi i32 [ %N, %while.cond1.preheader.us ], [ [[LOOP_DEC:%[^ ]+]], %while.body3.us ]
; CHECK-REGDEC: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[REM]], i32 1)
; CHECK-REGDEC: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK-REGDEC: br i1 [[CMP]], label %while.body3.us, label %while.cond1.while.end_crit_edge.us

; CHECK-NESTED: [[LOOP_DEC1:%[^ ]+]] = call i1 @llvm.loop.decrement.i32(i32 1)
; CHECK-NESTED: br i1 [[LOOP_DEC1]], label %while.cond1.preheader.us, label %while.end7

; CHECK-GUARD: while.cond1.preheader.us:
; CHECK-GUARD:   call void @llvm.set.loop.iterations.i32(i32 %N)
; CHECK-GUARD:   br label %while.body3.us

define void @nested(i32* nocapture %A, i32 %N) {
entry:
  %cmp20 = icmp eq i32 %N, 0
  br i1 %cmp20, label %while.end7, label %while.cond1.preheader.us

while.cond1.preheader.us:
  %i.021.us = phi i32 [ %inc6.us, %while.cond1.while.end_crit_edge.us ], [ 0, %entry ]
  %mul.us = mul i32 %i.021.us, %N
  br label %while.body3.us

while.body3.us:
  %j.019.us = phi i32 [ 0, %while.cond1.preheader.us ], [ %inc.us, %while.body3.us ]
  %add.us = add i32 %j.019.us, %mul.us
  %arrayidx.us = getelementptr inbounds i32, i32* %A, i32 %add.us
  store i32 %add.us, i32* %arrayidx.us, align 4
  %inc.us = add nuw i32 %j.019.us, 1
  %exitcond = icmp eq i32 %inc.us, %N
  br i1 %exitcond, label %while.cond1.while.end_crit_edge.us, label %while.body3.us

while.cond1.while.end_crit_edge.us:
  %inc6.us = add nuw i32 %i.021.us, 1
  %exitcond23 = icmp eq i32 %inc6.us, %N
  br i1 %exitcond23, label %while.end7, label %while.cond1.preheader.us

while.end7:
  ret void
}
