; RUN: opt -mtriple=thumbv8.1m.main-none-none-eabi -hardware-loops %s -S -o - | FileCheck %s
; RUN: opt -mtriple=thumbv8.1m.main-none-none-eabi -hardware-loops -disable-arm-loloops=true %s -S -o - | FileCheck %s --check-prefix=DISABLED
; RUN: llc -mtriple=thumbv8.1m.main-none-none-eabi %s -o - | FileCheck %s --check-prefix=CHECK-LLC

; DISABLED-NOT: llvm.{{.*}}.loop.iterations
; DISABLED-NOT: llvm.loop.decrement

@g = common local_unnamed_addr global i32* null, align 4

; CHECK-LABEL: do_copy
; CHECK: call void @llvm.set.loop.iterations.i32(i32 %n)
; CHECK: br label %while.body

; CHECK: [[REM:%[^ ]+]] = phi i32 [ %n, %entry ], [ [[LOOP_DEC:%[^ ]+]], %while.body ]
; CHECK: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[REM]], i32 1)
; CHECK: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK: br i1 [[CMP]], label %while.body, label %while.end

; CHECK-LLC-LABEL:do_copy:
; CHECK-LLC-NOT:    mov lr, r0
; CHECK-LLC:        dls lr, r0
; CHECK-LLC-NOT:    mov lr, r0
; CHECK-LLC:      [[LOOP_HEADER:\.LBB[0-9_]+]]:
; CHECK-LLC:        le lr, [[LOOP_HEADER]]
; CHECK-LLC-NOT:    b [[LOOP_EXIT:\.LBB[0-9._]+]]
; CHECK-LLC:      @ %while.end
define i32 @do_copy(i32 %n, i32* nocapture %p, i32* nocapture readonly %q) {
entry:
  br label %while.body

while.body:
  %q.addr.05 = phi i32* [ %incdec.ptr, %while.body ], [ %q, %entry ]
  %p.addr.04 = phi i32* [ %incdec.ptr1, %while.body ], [ %p, %entry ]
  %x.addr.03 = phi i32 [ %dec, %while.body ], [ %n, %entry ]
  %dec = add nsw i32 %x.addr.03, -1
  %incdec.ptr = getelementptr inbounds i32, i32* %q.addr.05, i32 1
  %0 = load i32, i32* %q.addr.05, align 4
  %incdec.ptr1 = getelementptr inbounds i32, i32* %p.addr.04, i32 1
  store i32 %0, i32* %p.addr.04, align 4
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:
  ret i32 0
}

; CHECK-LABEL: do_inc1
; CHECK: entry:
; CHECK: [[TEST:%[^ ]+]] = call i1 @llvm.test.set.loop.iterations.i32(i32 %n)
; CHECK: br i1 [[TEST]], label %while.body.lr.ph, label %while.end

; CHECK: while.body.lr.ph:
; CHECK: br label %while.body

; CHECK: [[REM:%[^ ]+]] = phi i32 [ %n, %while.body.lr.ph ], [ [[LOOP_DEC:%[^ ]+]], %while.body ]
; CHECK: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[REM]], i32 1)
; CHECK: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK: br i1 [[CMP]], label %while.body, label %while.end.loopexit

; CHECK-LLC-LABEL:do_inc1:
; CHECK-LLC:        wls lr, {{.*}}, [[LOOP_EXIT:.[LBB_0-3]+]]
; CHECK-LLC-NOT:    mov lr,
; CHECK-LLC:      [[LOOP_HEADER:\.LBB[0-9_]+]]:
; CHECK-LLC:        le lr, [[LOOP_HEADER]]
; CHECK-LLC-NOT:    b [[LOOP_EXIT:\.LBB[0-9_]+]]
; CHECK-LLC:      [[LOOP_EXIT]]:

define i32 @do_inc1(i32 %n) {
entry:
  %cmp7 = icmp eq i32 %n, 0
  br i1 %cmp7, label %while.end, label %while.body.lr.ph

while.body.lr.ph:
  %0 = load i32*, i32** @g, align 4
  br label %while.body

while.body:
  %i.09 = phi i32 [ 0, %while.body.lr.ph ], [ %inc1, %while.body ]
  %res.08 = phi i32 [ 0, %while.body.lr.ph ], [ %add, %while.body ]
  %arrayidx = getelementptr inbounds i32, i32* %0, i32 %i.09
  %1 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %1, %res.08
  %inc1 = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc1, %n
  br i1 %exitcond, label %while.end.loopexit, label %while.body

while.end.loopexit:
  br label %while.end

while.end:
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %while.end.loopexit ]
  ret i32 %res.0.lcssa
}

; CHECK-LABEL: do_inc2
; CHECK: entry:
; CHECK: [[ROUND:%[^ ]+]] = add i32 %n, -1
; CHECK: [[HALVE:%[^ ]+]] = lshr i32 [[ROUND]], 1
; CHECK: [[COUNT:%[^ ]+]] = add nuw i32 [[HALVE]], 1

; CHECK: while.body.lr.ph:
; CHECK:   call void @llvm.set.loop.iterations.i32(i32 [[COUNT]])
; CHECK:   br label %while.body
; CHECK: while.body:
; CHECK:   [[REM:%[^ ]+]] = phi i32 [ [[COUNT]], %while.body.lr.ph ], [ [[LOOP_DEC:%[^ ]+]], %while.body ]
; CHECK:   [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[REM]], i32 1)
; CHECK:   [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK:   br i1 [[CMP]], label %while.body, label %while.end.loopexit

; CHECK-LLC:      do_inc2:
; CHECK-LLC-NOT:    mov lr,
; CHECK-LLC:        dls lr, {{.*}}
; CHECK-LLC-NOT:    mov lr,
; CHECK-LLC:      [[LOOP_HEADER:\.LBB[0-9._]+]]:
; CHECK-LLC:        le lr, [[LOOP_HEADER]]

define i32 @do_inc2(i32 %n) {
entry:
  %cmp7 = icmp sgt i32 %n, 0
  br i1 %cmp7, label %while.body.lr.ph, label %while.end

while.body.lr.ph:
  %0 = load i32*, i32** @g, align 4
  br label %while.body

while.body:
  %i.09 = phi i32 [ 0, %while.body.lr.ph ], [ %add1, %while.body ]
  %res.08 = phi i32 [ 0, %while.body.lr.ph ], [ %add, %while.body ]
  %arrayidx = getelementptr inbounds i32, i32* %0, i32 %i.09
  %1 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %1, %res.08
  %add1 = add nuw nsw i32 %i.09, 2
  %cmp = icmp slt i32 %add1, %n
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:
  br label %while.end

while.end:
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %while.end.loopexit ]
  ret i32 %res.0.lcssa
}

; CHECK-LABEL: do_dec2

; CHECK: entry:
; CHECK: [[ROUND:%[^ ]+]] = add i32 %n, 1
; CHECK: [[CMP:%[^ ]+]] = icmp slt i32 %n, 2
; CHECK: [[SMIN:%[^ ]+]] = select i1 [[CMP]], i32 %n, i32 2
; CHECK: [[SUB:%[^ ]+]] = sub i32 [[ROUND]], [[SMIN]]
; CHECK: [[HALVE:%[^ ]+]] = lshr i32 [[SUB]], 1
; CHECK: [[COUNT:%[^ ]+]] = add nuw i32 [[HALVE]], 1

; CHECK: while.body.lr.ph:
; CHECK: call void @llvm.set.loop.iterations.i32(i32 [[COUNT]])
; CHECK: br label %while.body

; CHECK: [[REM:%[^ ]+]] = phi i32 [ [[COUNT]], %while.body.lr.ph ], [ [[LOOP_DEC:%[^ ]+]], %while.body ]
; CHECK: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[REM]], i32 1)
; CHECK: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK: br i1 [[CMP]], label %while.body, label %while.end.loopexit

; CHECK-LLC:      do_dec2
; CHECK-LLC-NOT:    mov lr,
; CHECK-LLC:        dls lr, {{.*}}
; CHECK-LLC-NOT:    mov lr,
; CHECK-LLC:      [[LOOP_HEADER:\.LBB[0-9_]+]]:
; CHECK-LLC:        le lr, [[LOOP_HEADER]]
; CHECK-LLC-NOT:    b .
define i32 @do_dec2(i32 %n) {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %while.body.lr.ph, label %while.end

while.body.lr.ph:
  %0 = load i32*, i32** @g, align 4
  br label %while.body

while.body:
  %i.08 = phi i32 [ %n, %while.body.lr.ph ], [ %sub, %while.body ]
  %res.07 = phi i32 [ 0, %while.body.lr.ph ], [ %add, %while.body ]
  %arrayidx = getelementptr inbounds i32, i32* %0, i32 %i.08
  %1 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %1, %res.07
  %sub = add nsw i32 %i.08, -2
  %cmp = icmp sgt i32 %i.08, 2
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:
  br label %while.end

while.end:
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %while.end.loopexit ]
  ret i32 %res.0.lcssa
}
