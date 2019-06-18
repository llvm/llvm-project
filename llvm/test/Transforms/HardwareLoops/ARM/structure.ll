; RUN: opt -mtriple=thumbv8.1m.main-arm-none-eabi -hardware-loops -disable-arm-loloops=false %s -S -o - | FileCheck %s

; CHECK-LABEL: early_exit
; CHECK-NOT: llvm.set.loop.iterations
; CHECK-NOT: llvm.loop.decrement
define i32 @early_exit(i32* nocapture readonly %a, i32 %max, i32 %n) {
entry:
  br label %do.body

do.body:
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %if.end ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %i.0
  %0 = load i32, i32* %arrayidx, align 4
  %cmp = icmp sgt i32 %0, %max
  br i1 %cmp, label %do.end, label %if.end

if.end:
  %inc = add nuw i32 %i.0, 1
  %cmp1 = icmp ult i32 %inc, %n
  br i1 %cmp1, label %do.body, label %if.end.do.end_crit_edge

if.end.do.end_crit_edge:
  %arrayidx2.phi.trans.insert = getelementptr inbounds i32, i32* %a, i32 %inc
  %.pre = load i32, i32* %arrayidx2.phi.trans.insert, align 4
  br label %do.end

do.end:
  %1 = phi i32 [ %.pre, %if.end.do.end_crit_edge ], [ %0, %do.body ]
  ret i32 %1
}

; CHECK-LABEL: nested
; CHECK-NOT: call void @llvm.set.loop.iterations.i32(i32 %N)
; CHECK: br i1 %cmp20, label %while.end7, label %while.cond1.preheader.us

; CHECK: call void @llvm.set.loop.iterations.i32(i32 %N)
; CHECK: br label %while.body3.us

; CHECK: [[REM:%[^ ]+]] = phi i32 [ %N, %while.cond1.preheader.us ], [ [[LOOP_DEC:%[^ ]+]], %while.body3.us ]
; CHECK: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 [[REM]], i32 1)
; CHECK: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK: br i1 [[CMP]], label %while.body3.us, label %while.cond1.while.end_crit_edge.us

; CHECK-NOT: [[LOOP_DEC1:%[^ ]+]] = call i1 @llvm.loop.decrement.i32(i32 1)
; CHECK-NOT: br i1 [[LOOP_DEC1]], label %while.cond1.preheader.us, label %while.end7
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

; CHECK-LABEL: pre_existing
; CHECK: llvm.set.loop.iterations
; CHECK-NOT: llvm.set.loop.iterations
; CHECK: call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %0, i32 1)
; CHECK-NOT: call i32 @llvm.loop.decrement.reg
define i32 @pre_existing(i32 %n, i32* nocapture %p, i32* nocapture readonly %q) {
entry:
  call void @llvm.set.loop.iterations.i32(i32 %n)
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %q.addr.05 = phi i32* [ %incdec.ptr, %while.body ], [ %q, %entry ]
  %p.addr.04 = phi i32* [ %incdec.ptr1, %while.body ], [ %p, %entry ]
  %0 = phi i32 [ %n, %entry ], [ %2, %while.body ]
  %incdec.ptr = getelementptr inbounds i32, i32* %q.addr.05, i32 1
  %1 = load i32, i32* %q.addr.05, align 4
  %incdec.ptr1 = getelementptr inbounds i32, i32* %p.addr.04, i32 1
  store i32 %1, i32* %p.addr.04, align 4
  %2 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %0, i32 1)
  %3 = icmp ne i32 %2, 0
  br i1 %3, label %while.body, label %while.end

while.end:                                        ; preds = %while.body
  ret i32 0
}

; CHECK-LABEL: pre_existing_inner
; CHECK-NOT: llvm.set.loop.iterations
; CHECK: while.cond1.preheader.us:
; CHECK: call void @llvm.set.loop.iterations.i32(i32 %N)
; CHECK: call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %0, i32 1)
; CHECK: br i1
; CHECK-NOT: call i32 @llvm.loop.decrement
define void @pre_existing_inner(i32* nocapture %A, i32 %N) {
entry:
  %cmp20 = icmp eq i32 %N, 0
  br i1 %cmp20, label %while.end7, label %while.cond1.preheader.us

while.cond1.preheader.us:
  %i.021.us = phi i32 [ %inc6.us, %while.cond1.while.end_crit_edge.us ], [ 0, %entry ]
  %mul.us = mul i32 %i.021.us, %N
  call void @llvm.set.loop.iterations.i32(i32 %N)
  br label %while.body3.us

while.body3.us:
  %j.019.us = phi i32 [ 0, %while.cond1.preheader.us ], [ %inc.us, %while.body3.us ]
  %0 = phi i32 [ %N, %while.cond1.preheader.us ], [ %1, %while.body3.us ]
  %add.us = add i32 %j.019.us, %mul.us
  %arrayidx.us = getelementptr inbounds i32, i32* %A, i32 %add.us
  store i32 %add.us, i32* %arrayidx.us, align 4
  %inc.us = add nuw i32 %j.019.us, 1
  %1 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %0, i32 1)
  %2 = icmp ne i32 %1, 0
  br i1 %2, label %while.body3.us, label %while.cond1.while.end_crit_edge.us

while.cond1.while.end_crit_edge.us:
  %inc6.us = add nuw i32 %i.021.us, 1
  %exitcond23 = icmp eq i32 %inc6.us, %N
  br i1 %exitcond23, label %while.end7, label %while.cond1.preheader.us

while.end7:
  ret void
}

; CHECK-LABEL: not_rotated
; CHECK-NOT: call void @llvm.set.loop.iterations
; CHECK-NOT: call i32 @llvm.loop.decrement.i32
define void @not_rotated(i32, i16* nocapture, i16 signext) {
  br label %4

4:
  %5 = phi i32 [ 0, %3 ], [ %19, %18 ]
  %6 = icmp eq i32 %5, %0
  br i1 %6, label %20, label %7

7:
  %8 = mul i32 %5, %0
  br label %9

9:
  %10 = phi i32 [ %17, %12 ], [ 0, %7 ]
  %11 = icmp eq i32 %10, %0
  br i1 %11, label %18, label %12

12:
  %13 = add i32 %10, %8
  %14 = getelementptr inbounds i16, i16* %1, i32 %13
  %15 = load i16, i16* %14, align 2
  %16 = add i16 %15, %2
  store i16 %16, i16* %14, align 2
  %17 = add i32 %10, 1
  br label %9

18:
  %19 = add i32 %5, 1
  br label %4

20:
  ret void
}

; CHECK-LABEL: multi_latch
; CHECK-NOT: call void @llvm.set.loop.iterations
; CHECK-NOT: call i32 @llvm.loop.decrement
define void @multi_latch(i32* %a, i32* %b, i32 %N) {
entry:
  %half = lshr i32 %N, 1
  br label %header

header:
  %iv = phi i32 [ 0, %entry ], [ %count.next, %latch.0 ], [ %count.next, %latch.1 ]
  %cmp = icmp ult i32 %iv, %half
  %addr.a = getelementptr i32, i32* %a, i32 %iv
  %addr.b = getelementptr i32, i32* %b, i32 %iv
  br i1 %cmp, label %if.then, label %if.else

if.then:
  store i32 %iv, i32* %addr.a
  br label %latch.0

if.else:
  store i32 %iv, i32* %addr.b
  br label %latch.0

latch.0:
  %count.next = add nuw i32 %iv, 1
  %cmp.1 = icmp ult i32 %count.next, %half
  br i1 %cmp.1, label %header, label %latch.1

latch.1:
  %ld = load i32, i32* %addr.a
  store i32 %ld, i32* %addr.b
  %cmp.2 = icmp ult i32 %count.next, %N
  br i1 %cmp.2, label %header, label %latch.1

exit:
  ret void
}


declare void @llvm.set.loop.iterations.i32(i32) #0
declare i32 @llvm.loop.decrement.reg.i32.i32.i32(i32, i32) #0

