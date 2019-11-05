; RUN: opt -mtriple=thumbv8.1m.main-arm-none-eabi -hardware-loops %s -S -o - | \
; RUN:     FileCheck %s
; RUN: llc -mtriple=thumbv8.1m.main-arm-none-eabi %s -o - | \
; RUN:     FileCheck %s --check-prefix=CHECK-LLC
; RUN: opt -mtriple=thumbv8.1m.main -loop-unroll -unroll-remainder=false -S < %s | \
; RUN:     llc -mtriple=thumbv8.1m.main | FileCheck %s --check-prefix=CHECK-UNROLL
; RUN: opt -mtriple=thumbv8.1m.main-arm-none-eabi -hardware-loops \
; RUN:     -pass-remarks-analysis=hardware-loops  %s -S -o - 2>&1 | \
; RUN:     FileCheck %s --check-prefix=CHECK-REMARKS


; CHECK-REMARKS: remark: <unknown>:0:0: hardware-loop not created: it's not profitable to create a hardware-loop
; CHECK-REMARKS: remark: <unknown>:0:0: hardware-loop not created: nested hardware-loops not supported
; CHECK-REMARKS: remark: <unknown>:0:0: hardware-loop not created: it's not profitable to create a hardware-loop
; CHECK-REMARKS: remark: <unknown>:0:0: hardware-loop not created: it's not profitable to create a hardware-loop
; CHECK-REMARKS: remark: <unknown>:0:0: hardware-loop not created: it's not profitable to create a hardware-loop
; CHECK-REMARKS: remark: <unknown>:0:0: hardware-loop not created: it's not profitable to create a hardware-loop
; CHECK-REMARKS: remark: <unknown>:0:0: hardware-loop not created: loop is not a candidate
; CHECK-REMARKS: remark: <unknown>:0:0: hardware-loop not created: nested hardware-loops not supported
; CHECK-REMARKS: remark: <unknown>:0:0: hardware-loop not created: it's not profitable to create a hardware-loop
; CHECK-REMARKS: remark: <unknown>:0:0: hardware-loop not created: it's not profitable to create a hardware-loop


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

; CHECK-LLC:      nested:
; CHECK-LLC-NOT:    mov lr, r1
; CHECK-LLC:        dls lr, r1
; CHECK-LLC-NOT:    mov lr, r1
; CHECK-LLC:      [[LOOP_HEADER:\.LBB[0-9._]+]]:
; CHECK-LLC:        le lr, [[LOOP_HEADER]]
; CHECK-LLC-NOT:    b [[LOOP_EXIT:\.LBB[0-9._]+]]
; CHECK-LLC:      [[LOOP_EXIT:\.LBB[0-9._]+]]:

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

; CHECK-LABEL: pre_existing_test_set
; CHECK: call i1 @llvm.test.set.loop.iterations
; CHECK-NOT: llvm.set{{.*}}.loop.iterations
; CHECK: call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %0, i32 1)
; CHECK-NOT: call i32 @llvm.loop.decrement.reg
define i32 @pre_existing_test_set(i32 %n, i32* nocapture %p, i32* nocapture readonly %q) {
entry:
  %guard = call i1 @llvm.test.set.loop.iterations.i32(i32 %n)
  br i1 %guard, label %while.preheader, label %while.end

while.preheader:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %q.addr.05 = phi i32* [ %incdec.ptr, %while.body ], [ %q, %while.preheader ]
  %p.addr.04 = phi i32* [ %incdec.ptr1, %while.body ], [ %p, %while.preheader ]
  %0 = phi i32 [ %n, %while.preheader ], [ %2, %while.body ]
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

; CHECK-LABEL: search
; CHECK: entry:
; CHECK:   [[TEST:%[^ ]+]] = call i1 @llvm.test.set.loop.iterations.i32(i32 %N)
; CHECK:   br i1 [[TEST]], label %for.body.preheader, label %for.cond.cleanup
; CHECK: for.body.preheader:
; CHECK:   br label %for.body
; CHECK: for.body:
; CHECK: for.inc:
; CHECK:   [[LOOP_DEC:%[^ ]+]] = call i32 @llvm.loop.decrement.reg.i32.i32.i32
; CHECK:   [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK:   br i1 [[CMP]], label %for.body, label %for.cond.cleanup
define i32 @search(i8* nocapture readonly %c, i32 %N) {
entry:
  %cmp11 = icmp eq i32 %N, 0
  br i1 %cmp11, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  %found.0.lcssa = phi i32 [ 0, %entry ], [ %found.1, %for.inc ]
  %spaces.0.lcssa = phi i32 [ 0, %entry ], [ %spaces.1, %for.inc ]
  %sub = sub nsw i32 %found.0.lcssa, %spaces.0.lcssa
  ret i32 %sub

for.body:
  %i.014 = phi i32 [ %inc3, %for.inc ], [ 0, %entry ]
  %spaces.013 = phi i32 [ %spaces.1, %for.inc ], [ 0, %entry ]
  %found.012 = phi i32 [ %found.1, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, i8* %c, i32 %i.014
  %0 = load i8, i8* %arrayidx, align 1
  switch i8 %0, label %for.inc [
    i8 108, label %sw.bb
    i8 111, label %sw.bb
    i8 112, label %sw.bb
    i8 32, label %sw.bb1
  ]

sw.bb:                                            ; preds = %for.body, %for.body, %for.body
  %inc = add nsw i32 %found.012, 1
  br label %for.inc

sw.bb1:                                           ; preds = %for.body
  %inc2 = add nsw i32 %spaces.013, 1
  br label %for.inc

for.inc:                                          ; preds = %sw.bb, %sw.bb1, %for.body
  %found.1 = phi i32 [ %found.012, %for.body ], [ %found.012, %sw.bb1 ], [ %inc, %sw.bb ]
  %spaces.1 = phi i32 [ %spaces.013, %for.body ], [ %inc2, %sw.bb1 ], [ %spaces.013, %sw.bb ]
  %inc3 = add nuw i32 %i.014, 1
  %exitcond = icmp eq i32 %inc3, %N
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: unroll_inc_int
; CHECK: call void @llvm.set.loop.iterations.i32(i32 %N)
; CHECK: call i32 @llvm.loop.decrement.reg.i32.i32.i32(

; TODO: We should be able to support the unrolled loop body.
; CHECK-UNROLL-LABEL: unroll_inc_int
; CHECK-UNROLL:     [[PREHEADER:.LBB[0-9_]+]]: @ %for.body.preheader
; CHECK-UNROLL-NOT: dls
; CHECK-UNROLL:     [[LOOP:.LBB[0-9_]+]]: @ %for.body
; CHECK-UNROLL-NOT: le lr, [[LOOP]]
; CHECK-UNROLL:     bne [[LOOP]]
; CHECK-UNROLL:     wls lr, lr, [[EXIT:.LBB[0-9_]+]]
; CHECK-UNROLL:     [[EPIL:.LBB[0-9_]+]]:
; CHECK-UNROLL:     le lr, [[EPIL]]
; CHECK-UNROLL-NEXT: [[EXIT]]

define void @unroll_inc_int(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %cmp8 = icmp sgt i32 %N, 0
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %c, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %mul = mul nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i32 %i.09
  store i32 %mul, i32* %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: unroll_inc_unsigned
; CHECK: call i1 @llvm.test.set.loop.iterations.i32(i32 %N)
; CHECK: call i32 @llvm.loop.decrement.reg.i32.i32.i32(

; CHECK-LLC-LABEL: unroll_inc_unsigned:
; CHECK-LLC: wls lr, r3, [[EXIT:.LBB[0-9_]+]]
; CHECK-LLC: [[HEADER:.LBB[0-9_]+]]:
; CHECK-LLC: le lr, [[HEADER]]
; CHECK-LLC-NEXT: [[EXIT]]:

; TODO: We should be able to support the unrolled loop body.
; CHECK-UNROLL-LABEL: unroll_inc_unsigned
; CHECK-UNROLL:     [[PREHEADER:.LBB[0-9_]+]]: @ %for.body.preheader
; CHECK-UNROLL-NOT: dls
; CHECK-UNROLL:     [[LOOP:.LBB[0-9_]+]]: @ %for.body
; CHECK-UNROLL-NOT: le lr, [[LOOP]]
; CHECK-UNROLL:     bne [[LOOP]]
; CHECK-UNROLL:     wls lr, lr, [[EPIL_EXIT:.LBB[0-9_]+]]
; CHECK-UNROLL: [[EPIL:.LBB[0-9_]+]]:
; CHECK-UNROLL:     le lr, [[EPIL]]
; CHECK-UNROLL: [[EPIL_EXIT]]:
; CHECK-UNROLL:     pop
define void @unroll_inc_unsigned(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %cmp8 = icmp eq i32 %N, 0
  br i1 %cmp8, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %c, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %mul = mul nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i32 %i.09
  store i32 %mul, i32* %arrayidx2, align 4
  %inc = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: unroll_dec_int
; CHECK: call void @llvm.set.loop.iterations.i32(i32 %N)
; CHECK: call i32 @llvm.loop.decrement.reg.i32.i32.i32(

; TODO: An unnecessary register is being held to hold COUNT, lr should just
; be used instead.
; CHECK-LLC-LABEL: unroll_dec_int:
; CHECK-LLC: dls lr, r3
; CHECK-LLC-NOT: mov lr, r3
; CHECK-LLC: [[HEADER:.LBB[0-9_]+]]:
; CHECK-LLC: le lr, [[HEADER]]

; CHECK-UNROLL-LABEL: unroll_dec_int:
; CHECK-UNROLL:         wls lr, {{.*}}, [[PROLOGUE_EXIT:.LBB[0-9_]+]]
; CHECK-UNROLL-NEXT: [[PROLOGUE:.LBB[0-9_]+]]:
; CHECK-UNROLL:         le lr, [[PROLOGUE]]
; CHECK-UNROLL-NEXT: [[PROLOGUE_EXIT:.LBB[0-9_]+]]:
; CHECK-UNROLL:         dls lr, lr
; CHECK-UNROLL:      [[BODY:.LBB[0-9_]+]]:
; CHECK-UNROLL:         le lr, [[BODY]]
; CHECK-UNROLL-NOT:     b
; CHECK-UNROLL:         pop
define void @unroll_dec_int(i32* nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %c, i32 %N) {
entry:
  %cmp8 = icmp sgt i32 %N, 0
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ %dec, %for.body ], [ %N, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %c, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %mul = mul nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i32 %i.09
  store i32 %mul, i32* %arrayidx2, align 4
  %dec = add nsw i32 %i.09, -1
  %cmp = icmp sgt i32 %dec, 0
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

declare void @llvm.set.loop.iterations.i32(i32) #0
declare i1 @llvm.test.set.loop.iterations.i32(i32) #0
declare i32 @llvm.loop.decrement.reg.i32.i32.i32(i32, i32) #0

