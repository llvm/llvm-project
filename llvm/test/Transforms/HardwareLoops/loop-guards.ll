; RUN: opt -passes='hardware-loops<force-hardware-loops;force-hardware-loop-guard;hardware-loop-decrement=1;hardware-loop-counter-bitwidth=32>' -S %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-EXIT
; RUN: opt -passes='hardware-loops<force-hardware-loops;force-hardware-loop-guard;force-hardware-loop-phi;hardware-loop-decrement=1;hardware-loop-counter-bitwidth=32>' -S %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-LATCH
; RUN: opt -passes='hardware-loops<force-hardware-loops;hardware-loop-decrement=1;hardware-loop-counter-bitwidth=32>' -S %s -o - | FileCheck %s --check-prefix=NO-GUARD

; NO-GUARD-NOT: @llvm.test.set.loop.iterations

; CHECK-LABEL: test1
; CHECK: entry:
; CHECK:   [[MAX:%[^ ]+]] = call i32 @llvm.umax.i32(i32 %N, i32 2)
; CHECK:   [[COUNT:%[^ ]+]] = add i32 [[MAX]], -1
; CHECK:   br i1 %t1, label %do.body.preheader
; CHECK: do.body.preheader:
; CHECK-EXIT:   call void @llvm.set.loop.iterations.i32(i32 [[COUNT]])
; CHECK-LATCH:   call i32 @llvm.start.loop.iterations.i32(i32 [[COUNT]])
; CHECK:   br label %do.body
define void @test1(i1 zeroext %t1, ptr nocapture %a, ptr nocapture readonly %b, i32 %N) {
entry:
  br i1 %t1, label %do.body, label %if.end

do.body:                                          ; preds = %do.body, %entry
  %b.addr.0 = phi ptr [ %incdec.ptr, %do.body ], [ %b, %entry ]
  %a.addr.0 = phi ptr [ %incdec.ptr1, %do.body ], [ %a, %entry ]
  %i.0 = phi i32 [ %inc, %do.body ], [ 1, %entry ]
  %incdec.ptr = getelementptr inbounds i32, ptr %b.addr.0, i32 1
  %tmp = load i32, ptr %b.addr.0, align 4
  %incdec.ptr1 = getelementptr inbounds i32, ptr %a.addr.0, i32 1
  store i32 %tmp, ptr %a.addr.0, align 4
  %inc = add nuw i32 %i.0, 1
  %cmp = icmp ult i32 %inc, %N
  br i1 %cmp, label %do.body, label %if.end

if.end:                                           ; preds = %do.body, %entry
  ret void
}

; CHECK-LABEL: test2
; CHECK-NOT: call i1 @llvm.test.set.loop.iterations
; CHECK-NOT: call void @llvm.set.loop.iterations
; CHECK-NOT: call i32 @llvm.start.loop.iterations
define void @test2(i1 zeroext %t1, ptr nocapture %a, ptr nocapture readonly %b, i32 %N) {
entry:
  br i1 %t1, label %do.body, label %if.end

do.body:                                          ; preds = %do.body, %entry
  %b.addr.0 = phi ptr [ %incdec.ptr, %do.body ], [ %b, %entry ]
  %a.addr.0 = phi ptr [ %incdec.ptr1, %do.body ], [ %a, %entry ]
  %i.0 = phi i32 [ %add, %do.body ], [ 1, %entry ]
  %incdec.ptr = getelementptr inbounds i32, ptr %b.addr.0, i32 1
  %tmp = load i32, ptr %b.addr.0, align 4
  %incdec.ptr1 = getelementptr inbounds i32, ptr %a.addr.0, i32 1
  store i32 %tmp, ptr %a.addr.0, align 4
  %add = add i32 %i.0, 2
  %cmp = icmp ult i32 %add, %N
  br i1 %cmp, label %do.body, label %if.end

if.end:                                           ; preds = %do.body, %entry
  ret void
}

; CHECK-LABEL: test3
; CHECK: entry:
; CHECK:   [[COUNT:%[^ ]+]] = call i32 @llvm.umax.i32(i32 %N, i32 1)
; CHECK:   br i1 %brmerge.demorgan, label %do.body.preheader
; CHECK: do.body.preheader:
; CHECK-EXIT:   call void @llvm.set.loop.iterations.i32(i32 [[COUNT]])
; CHECK-LATCH:   call i32 @llvm.start.loop.iterations.i32(i32 [[COUNT]])
; CHECK:   br label %do.body
define void @test3(i1 zeroext %t1, i1 zeroext %t2, ptr nocapture %a, ptr nocapture readonly %b, i32 %N) {
entry:
  %brmerge.demorgan = and i1 %t1, %t2
  br i1 %brmerge.demorgan, label %do.body, label %if.end

do.body:                                          ; preds = %do.body, %entry
  %b.addr.0 = phi ptr [ %incdec.ptr, %do.body ], [ %b, %entry ]
  %a.addr.0 = phi ptr [ %incdec.ptr3, %do.body ], [ %a, %entry ]
  %i.0 = phi i32 [ %inc, %do.body ], [ 0, %entry ]
  %incdec.ptr = getelementptr inbounds i32, ptr %b.addr.0, i32 1
  %tmp = load i32, ptr %b.addr.0, align 4
  %incdec.ptr3 = getelementptr inbounds i32, ptr %a.addr.0, i32 1
  store i32 %tmp, ptr %a.addr.0, align 4
  %inc = add nuw i32 %i.0, 1
  %cmp = icmp ult i32 %inc, %N
  br i1 %cmp, label %do.body, label %if.end

if.end:                                           ; preds = %do.body, %entry
  ret void
}

; CHECK-LABEL: test4
; CHECK: entry:
; CHECK-LATCH:  br i1 %brmerge.demorgan, label %while.cond
; CHECK-LATCH-NOT: @llvm{{.*}}loop.iterations 
; CHECK-EXIT:   br i1 %brmerge.demorgan, label %while.cond.preheader
; CHECK-EXIT: while.cond.preheader:
; CHECK-EXIT:   [[COUNT:%[^ ]+]] = add i32 %N, 1
; CHECK-EXIT:   call void @llvm.set.loop.iterations.i32(i32 [[COUNT]])
; CHECK-EXIT:   br label %while.cond
define void @test4(i1 zeroext %t1, i1 zeroext %t2, ptr nocapture %a, ptr nocapture readonly %b, i32 %N) {
entry:
  %brmerge.demorgan = and i1 %t1, %t2
  br i1 %brmerge.demorgan, label %while.cond, label %if.end

while.cond:                                       ; preds = %while.body, %entry
  %b.addr.0 = phi ptr [ %incdec.ptr, %while.body ], [ %b, %entry ]
  %a.addr.0 = phi ptr [ %incdec.ptr3, %while.body ], [ %a, %entry ]
  %i.0 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %exitcond = icmp eq i32 %i.0, %N
  br i1 %exitcond, label %if.end, label %while.body

while.body:                                       ; preds = %while.cond
  %incdec.ptr = getelementptr inbounds i32, ptr %b.addr.0, i32 1
  %tmp = load i32, ptr %b.addr.0, align 4
  %incdec.ptr3 = getelementptr inbounds i32, ptr %a.addr.0, i32 1
  store i32 %tmp, ptr %a.addr.0, align 4
  %inc = add i32 %i.0, 1
  br label %while.cond

if.end:                                           ; preds = %while.cond, %entry
  ret void
}

; CHECK-LABEL: test5
; CHECK: entry:
; CHECK:   br i1 %or.cond, label %while.body.preheader
; CHECK: while.body.preheader:
; CHECK-EXIT:   call void @llvm.set.loop.iterations.i32(i32 %N)
; CHECK-LATCH:   call i32 @llvm.start.loop.iterations.i32(i32 %N)
; CHECK:   br label %while.body
define void @test5(i1 zeroext %t1, i1 zeroext %t2, ptr nocapture %a, ptr nocapture readonly %b, i32 %N) {
entry:
  %brmerge.demorgan = and i1 %t1, %t2
  %cmp6 = icmp ne i32 %N, 0
  %or.cond = and i1 %brmerge.demorgan, %cmp6
  br i1 %or.cond, label %while.body, label %if.end

while.body:                                       ; preds = %while.body, %entry
  %i.09 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %a.addr.08 = phi ptr [ %incdec.ptr3, %while.body ], [ %a, %entry ]
  %b.addr.07 = phi ptr [ %incdec.ptr, %while.body ], [ %b, %entry ]
  %incdec.ptr = getelementptr inbounds i32, ptr %b.addr.07, i32 1
  %tmp = load i32, ptr %b.addr.07, align 4
  %incdec.ptr3 = getelementptr inbounds i32, ptr %a.addr.08, i32 1
  store i32 %tmp, ptr %a.addr.08, align 4
  %inc = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %if.end, label %while.body

if.end:                                           ; preds = %while.body, %entry
  ret void
}

; CHECK-LABEL: test6
; CHECK: entry:
; CHECK:   br i1 %brmerge.demorgan, label %while.preheader
; CHECK: while.preheader:
; CHECK-EXIT:   [[TEST:%[^ ]+]] = call i1 @llvm.test.set.loop.iterations.i32(i32 %N)
; CHECK-LATCH:   [[TEST1:%[^ ]+]] = call { i32, i1 } @llvm.test.start.loop.iterations.i32(i32 %N)
; CHECK-LATCH:  [[TEST:%[^ ]+]] = extractvalue { i32, i1 } [[TEST1]], 1
; CHECK:   br i1 [[TEST]], label %while.body.preheader, label %if.end
; CHECK: while.body.preheader:
; CHECK:   br label %while.body
define void @test6(i1 zeroext %t1, i1 zeroext %t2, ptr nocapture %a, ptr nocapture readonly %b, i32 %N) {
entry:
  %brmerge.demorgan = and i1 %t1, %t2
  br i1 %brmerge.demorgan, label %while.preheader, label %if.end

while.preheader:                                  ; preds = %entry
  %cmp = icmp ne i32 %N, 0
  br i1 %cmp, label %while.body, label %if.end

while.body:                                       ; preds = %while.body, %while.preheader
  %i.09 = phi i32 [ %inc, %while.body ], [ 0, %while.preheader ]
  %a.addr.08 = phi ptr [ %incdec.ptr3, %while.body ], [ %a, %while.preheader ]
  %b.addr.07 = phi ptr [ %incdec.ptr, %while.body ], [ %b, %while.preheader ]
  %incdec.ptr = getelementptr inbounds i32, ptr %b.addr.07, i32 1
  %tmp = load i32, ptr %b.addr.07, align 4
  %incdec.ptr3 = getelementptr inbounds i32, ptr %a.addr.08, i32 1
  store i32 %tmp, ptr %a.addr.08, align 4
  %inc = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %if.end, label %while.body

if.end:                                           ; preds = %while.body, %while.preheader, %entry
  ret void
}

; CHECK-LABEL: test7
; CHECK: entry:
; CHECK:   br i1 %brmerge.demorgan, label %while.preheader
; CHECK: while.preheader:
; CHECK-EXIT:   [[TEST:%[^ ]+]] = call i1 @llvm.test.set.loop.iterations.i32(i32 %N)
; CHECK-LATCH:   [[TEST1:%[^ ]+]] = call { i32, i1 } @llvm.test.start.loop.iterations.i32(i32 %N)
; CHECK-LATCH:  [[TEST:%[^ ]+]] = extractvalue { i32, i1 } [[TEST1]], 1
; CHECK:   br i1 [[TEST]], label %while.body.preheader, label %if.end
; CHECK: while.body.preheader:
; CHECK:   br label %while.body
define void @test7(i1 zeroext %t1, i1 zeroext %t2, ptr nocapture %a, ptr nocapture readonly %b, i32 %N) {
entry:
  %brmerge.demorgan = and i1 %t1, %t2
  br i1 %brmerge.demorgan, label %while.preheader, label %if.end

while.preheader:                                  ; preds = %entry
  %cmp = icmp eq i32 %N, 0
  br i1 %cmp, label %if.end, label %while.body

while.body:                                       ; preds = %while.body, %while.preheader
  %i.09 = phi i32 [ %inc, %while.body ], [ 0, %while.preheader ]
  %a.addr.08 = phi ptr [ %incdec.ptr3, %while.body ], [ %a, %while.preheader ]
  %b.addr.07 = phi ptr [ %incdec.ptr, %while.body ], [ %b, %while.preheader ]
  %incdec.ptr = getelementptr inbounds i32, ptr %b.addr.07, i32 1
  %tmp = load i32, ptr %b.addr.07, align 4
  %incdec.ptr3 = getelementptr inbounds i32, ptr %a.addr.08, i32 1
  store i32 %tmp, ptr %a.addr.08, align 4
  %inc = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %if.end, label %while.body

if.end:                                           ; preds = %while.body, %while.preheader, %entry
  ret void
}

; TODO: Can we rearrange the conditional blocks so that we can use the test form?
; CHECK-LABEL: test8
; CHECK: entry:
; CHECK:   [[CMP:%[^ ]+]] = icmp ne i32 %N, 0
; CHECK:   br i1 [[CMP]], label %while.preheader
; CHECK: while.preheader:
; CHECK:   br i1 %brmerge.demorgan, label %while.body.preheader
; CHECK: while.body.preheader:
; CHECK-EXIT:   call void @llvm.set.loop.iterations.i32(i32 %N)
; CHECK-LATCH:   call i32 @llvm.start.loop.iterations.i32(i32 %N)
; CHECK:   br label %while.body
define void @test8(i1 zeroext %t1, i1 zeroext %t2, ptr nocapture %a, ptr nocapture readonly %b, i32 %N) {
entry:
  %cmp = icmp ne i32 %N, 0
  br i1 %cmp, label %while.preheader, label %if.end

while.preheader:                                  ; preds = %entry
  %brmerge.demorgan = and i1 %t1, %t2
  br i1 %brmerge.demorgan, label %while.body, label %if.end

while.body:                                       ; preds = %while.body, %while.preheader
  %i.09 = phi i32 [ %inc, %while.body ], [ 0, %while.preheader ]
  %a.addr.08 = phi ptr [ %incdec.ptr3, %while.body ], [ %a, %while.preheader ]
  %b.addr.07 = phi ptr [ %incdec.ptr, %while.body ], [ %b, %while.preheader ]
  %incdec.ptr = getelementptr inbounds i32, ptr %b.addr.07, i32 1
  %tmp = load i32, ptr %b.addr.07, align 4
  %incdec.ptr3 = getelementptr inbounds i32, ptr %a.addr.08, i32 1
  store i32 %tmp, ptr %a.addr.08, align 4
  %inc = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %if.end, label %while.body

if.end:                                           ; preds = %while.body, %while.preheader, %entry
  ret void
}

; CHECK-LABEL: test9
; CHECK: entry:
; CHECK:   br i1 %brmerge.demorgan, label %do.body.preheader
; CHECK: do.body.preheader:
; CHECK-EXIT:   call void @llvm.set.loop.iterations.i32(i32 %N)
; CHECK-LATCH:   call i32 @llvm.start.loop.iterations.i32(i32 %N)
; CHECK:   br label %do.body
define void @test9(i1 zeroext %t1, ptr nocapture %a, ptr nocapture readonly %b, i32 %N) {
entry:
  %cmp = icmp ne i32 %N, 0
  %brmerge.demorgan = and i1 %t1, %cmp
  br i1 %brmerge.demorgan, label %do.body, label %if.end

do.body:                                          ; preds = %do.body, %entry
  %b.addr.0 = phi ptr [ %incdec.ptr, %do.body ], [ %b, %entry ]
  %a.addr.0 = phi ptr [ %incdec.ptr3, %do.body ], [ %a, %entry ]
  %i.0 = phi i32 [ %inc, %do.body ], [ 0, %entry ]
  %incdec.ptr = getelementptr inbounds i32, ptr %b.addr.0, i32 1
  %tmp = load i32, ptr %b.addr.0, align 4
  %incdec.ptr3 = getelementptr inbounds i32, ptr %a.addr.0, i32 1
  store i32 %tmp, ptr %a.addr.0, align 4
  %inc = add nuw i32 %i.0, 1
  %cmp.1 = icmp ult i32 %inc, %N
  br i1 %cmp.1, label %do.body, label %if.end

if.end:                                           ; preds = %do.body, %entry
  ret void
}

; CHECK-LABEL: test10
; CHECK: entry:
; CHECK:   br i1 %cmp.1, label %do.body.preheader
; CHECK: do.body.preheader:
; CHECK-EXIT:   call void @llvm.set.loop.iterations.i32(i32
; CHECK-LATCH:   call i32 @llvm.start.loop.iterations.i32(i32
; CHECK:   br label %do.body
define void @test10(ptr nocapture %a, ptr nocapture readonly %b, i32 %N) {
entry:
  %cmp = icmp ne i32 %N, 0
  %sub = sub i32 %N, 1
  %be = select i1 %cmp, i32 0, i32 %sub
  %cmp.1 = icmp ne i32 %be, 0
  br i1 %cmp.1, label %do.body, label %if.end

do.body:                                          ; preds = %do.body, %entry
  %b.addr.0 = phi ptr [ %incdec.ptr, %do.body ], [ %b, %entry ]
  %a.addr.0 = phi ptr [ %incdec.ptr3, %do.body ], [ %a, %entry ]
  %i.0 = phi i32 [ %inc, %do.body ], [ 0, %entry ]
  %incdec.ptr = getelementptr inbounds i32, ptr %b.addr.0, i32 1
  %tmp = load i32, ptr %b.addr.0, align 4
  %incdec.ptr3 = getelementptr inbounds i32, ptr %a.addr.0, i32 1
  store i32 %tmp, ptr %a.addr.0, align 4
  %inc = add nuw i32 %i.0, 1
  %cmp.2 = icmp ult i32 %inc, %N
  br i1 %cmp.2, label %do.body, label %if.end

if.end:                                           ; preds = %do.body, %entry
  ret void
}

; CHECK-LABEL: test11
; CHECK: entry:
; CHECK:   br label %do.body.preheader
; CHECK: do.body.preheader:
; CHECK-EXIT:   [[TEST:%[^ ]+]] = call i1 @llvm.test.set.loop.iterations.i32(i32 %N)
; CHECK-LATCH:  [[TEST1:%[^ ]+]] = call { i32, i1 } @llvm.test.start.loop.iterations.i32(i32 %N)
; CHECK-LATCH:  [[TEST:%[^ ]+]] = extractvalue { i32, i1 } [[TEST1]], 1
; CHECK:   br i1 [[TEST]], label %do.body.preheader1, label %if.end
; CHECK: do.body.preheader1:
; CHECK:   br label %do.body
define void @test11(i1 zeroext %t1, ptr nocapture %a, ptr nocapture readonly %b, i32 %N) {
entry:
  br label %do.body.preheader

do.body.preheader:
  %cmp = icmp ne i32 %N, 0
  br i1 %cmp, label %do.body, label %if.end

do.body:
  %b.addr.0 = phi ptr [ %incdec.ptr, %do.body ], [ %b, %do.body.preheader ]
  %a.addr.0 = phi ptr [ %incdec.ptr3, %do.body ], [ %a, %do.body.preheader ]
  %i.0 = phi i32 [ %inc, %do.body ], [ 0, %do.body.preheader ]
  %incdec.ptr = getelementptr inbounds i32, ptr %b.addr.0, i32 1
  %tmp = load i32, ptr %b.addr.0, align 4
  %incdec.ptr3 = getelementptr inbounds i32, ptr %a.addr.0, i32 1
  store i32 %tmp, ptr %a.addr.0, align 4
  %inc = add nuw i32 %i.0, 1
  %cmp.1 = icmp ult i32 %inc, %N
  br i1 %cmp.1, label %do.body, label %if.end

if.end:                                           ; preds = %do.body, %entry
  ret void
}

; CHECK-LABEL: test12
; CHECK: entry:
; CHECK-EXIT:   [[TEST:%[^ ]+]] = call i1 @llvm.test.set.loop.iterations.i32(i32 %conv)
; CHECK-LATCH:  [[TEST1:%[^ ]+]] = call { i32, i1 } @llvm.test.start.loop.iterations.i32(i32 %conv)
; CHECK-LATCH:  [[TEST:%[^ ]+]] = extractvalue { i32, i1 } [[TEST1]], 1
; CHECK:   br i1 [[TEST]], label %for.body.preheader, label %for.end
; CHECK: for.body.preheader:
; CHECK:   br label %for.body

define void @test12(ptr nocapture %a, ptr nocapture readonly %b, i16 zeroext %length) {
entry:
  %conv = zext i16 %length to i32
  %cmp8.not = icmp eq i16 %length, 0
  br i1 %cmp8.not, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i32 %i.09
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i32 %i.09
  store i32 %0, ptr %arrayidx2, align 4
  %inc = add nuw nsw i32 %i.09, 1
  %exitcond.not = icmp eq i32 %inc, %conv
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}
