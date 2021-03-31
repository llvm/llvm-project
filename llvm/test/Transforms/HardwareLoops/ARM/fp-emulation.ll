; RUN: opt -mtriple=thumbv8.1m.main-none-none-eabi -mattr=+fp-armv8 -hardware-loops %s -S -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-FP
; RUN: opt -mtriple=thumbv8.1m.main-none-none-eabi -mattr=+soft-float -hardware-loops %s -S -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-SOFT

; CHECK-LABEL: test_fptosi
; CHECK-SOFT-NOT: call i32 @llvm.start.loop.iterations

; CHECK: entry:
; CHECK-FP: [[COUNT:%[^ ]+]] = call i32 @llvm.umax.i32(i32 %n, i32 1)

; CHECK: while.body.lr.ph:
; CHECK-FP: [[START:%[^ ]+]] = call i32 @llvm.start.loop.iterations.i32(i32 [[COUNT]])
; CHECK-FP-NEXT: br label %while.body

; CHECK-FP: [[REM:%[^ ]+]] = phi i32 [ [[START]], %while.body.lr.ph ], [ [[LOOP_DEC:%[^ ]+]], %if.end4 ]
; CHECK-FP: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[REM]], i32 1)
; CHECK-FP: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK-FP: br i1 [[CMP]], label %while.body, label %cleanup.loopexit

define void @test_fptosi(i32 %n, i32** %g, double** %d) {
entry:
  %n.off = add i32 %n, -1
  %0 = icmp ult i32 %n.off, 500
  br i1 %0, label %while.body.lr.ph, label %cleanup

while.body.lr.ph:
  %1 = load double*, double** %d, align 4
  %2 = load i32*, i32** %g, align 4
  br label %while.body

while.body:
  %i.012 = phi i32 [ 0, %while.body.lr.ph ], [ %inc, %if.end4 ]
  %rem = urem i32 %i.012, 10
  %tobool = icmp eq i32 %rem, 0
  br i1 %tobool, label %if.end4, label %if.then2

if.then2:
  %arrayidx = getelementptr inbounds double, double* %1, i32 %i.012
  %3 = load double, double* %arrayidx, align 8
  %conv = fptosi double %3 to i32
  %arrayidx3 = getelementptr inbounds i32, i32* %2, i32 %i.012
  store i32 %conv, i32* %arrayidx3, align 4
  br label %if.end4

if.end4:
  %inc = add nuw i32 %i.012, 1
  %cmp1 = icmp ult i32 %inc, %n
  br i1 %cmp1, label %while.body, label %cleanup.loopexit

cleanup.loopexit:
  br label %cleanup

cleanup:
  ret void
}

; CHECK-LABEL: test_fptoui
; CHECK: entry:
; CHECK-FP: [[COUNT:%[^ ]+]] = call i32 @llvm.umax.i32(i32 %n, i32 1)
; CHECK-FP: while.body.lr.ph:
; CHECK-FP: [[START:%[^ ]+]] = call i32 @llvm.start.loop.iterations.i32(i32 [[COUNT]])
; CHECK-FP-NEXT: br label %while.body

; CHECK-FP: [[REM:%[^ ]+]] = phi i32 [ [[START]], %while.body.lr.ph ], [ [[LOOP_DEC:%[^ ]+]], %if.end4 ]
; CHECK-FP: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[REM]], i32 1)
; CHECK-FP: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK-FP: br i1 [[CMP]], label %while.body, label %cleanup.loopexit

; CHECK-SOFT-NOT: call i32 @llvm.start.loop.iterations

define void @test_fptoui(i32 %n, i32** %g, double** %d) {
entry:
  %n.off = add i32 %n, -1
  %0 = icmp ult i32 %n.off, 500
  br i1 %0, label %while.body.lr.ph, label %cleanup

while.body.lr.ph:
  %1 = load double*, double** %d, align 4
  %2 = load i32*, i32** %g, align 4
  br label %while.body

while.body:
  %i.012 = phi i32 [ 0, %while.body.lr.ph ], [ %inc, %if.end4 ]
  %rem = urem i32 %i.012, 10
  %tobool = icmp eq i32 %rem, 0
  br i1 %tobool, label %if.end4, label %if.then2

if.then2:
  %arrayidx = getelementptr inbounds double, double* %1, i32 %i.012
  %3 = load double, double* %arrayidx, align 8
  %conv = fptoui double %3 to i32
  %arrayidx3 = getelementptr inbounds i32, i32* %2, i32 %i.012
  store i32 %conv, i32* %arrayidx3, align 4
  br label %if.end4

if.end4:
  %inc = add nuw i32 %i.012, 1
  %cmp1 = icmp ult i32 %inc, %n
  br i1 %cmp1, label %while.body, label %cleanup.loopexit

cleanup.loopexit:
  br label %cleanup

cleanup:
  ret void
}

; CHECK-LABEL: load_store_float
; CHECK: entry:
; CHECK:   [[COUNT:%[^ ]+]] = call i32 @llvm.umax.i32(i32 %n, i32 1)
; CHECK: while.body.lr.ph:
; CHECK:   [[START:%[^ ]+]] = call i32 @llvm.start.loop.iterations.i32(i32 [[COUNT]])
; CHECK-NEXT: br label %while.body

; CHECK: [[REM:%[^ ]+]] = phi i32 [ [[START]], %while.body.lr.ph ], [ [[LOOP_DEC:%[^ ]+]], %if.end4 ]
; CHECK: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[REM]], i32 1)
; CHECK: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK: br i1 [[CMP]], label %while.body, label %cleanup.loopexit

define void @load_store_float(i32 %n, double** %d, double** %g) {
entry:
  %n.off = add i32 %n, -1
  %0 = icmp ult i32 %n.off, 500
  br i1 %0, label %while.body.lr.ph, label %cleanup

while.body.lr.ph:
  %1 = load double*, double** %d, align 4
  %2 = load double*, double** %g, align 4
  br label %while.body

while.body:
  %i.012 = phi i32 [ 0, %while.body.lr.ph ], [ %inc, %if.end4 ]
  %rem = urem i32 %i.012, 10
  %tobool = icmp eq i32 %rem, 0
  br i1 %tobool, label %if.end4, label %if.then2

if.then2:
  %arrayidx = getelementptr inbounds double, double* %1, i32 %i.012
  %3 = load double, double* %arrayidx, align 8
  %arrayidx3 = getelementptr inbounds double, double* %2, i32 %i.012
  store double %3, double* %arrayidx3, align 8
  br label %if.end4

if.end4:
  %inc = add nuw i32 %i.012, 1
  %cmp1 = icmp ult i32 %inc, %n
  br i1 %cmp1, label %while.body, label %cleanup.loopexit

cleanup.loopexit:
  br label %cleanup

cleanup:
  ret void
}

; CHECK-LABEL: fp_add
; CHECK-SOFT-NOT: call i32 @llvm.start.loop.iterations
; CHECK: entry:
; CHECK-FP: [[COUNT:%[^ ]+]] = call i32 @llvm.umax.i32(i32 %n, i32 1)
; CHECK: while.body.lr.ph:
; CHECK-FP: [[START:%[^ ]+]] = call i32 @llvm.start.loop.iterations.i32(i32 [[COUNT]])
; CHECK: br label %while.body

; CHECK-SOFT-NOT: call i32 @llvm.loop.decrement

; CHECK-FP: [[REM:%[^ ]+]] = phi i32 [ [[START]], %while.body.lr.ph ], [ [[LOOP_DEC:%[^ ]+]], %if.end4 ]
; CHECK-FP: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[REM]], i32 1)
; CHECK-FP: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK-FP: br i1 [[CMP]], label %while.body, label %cleanup.loopexit

define void @fp_add(i32 %n, float** %d, float** %g) {
entry:
  %n.off = add i32 %n, -1
  %0 = icmp ult i32 %n.off, 500
  br i1 %0, label %while.body.lr.ph, label %cleanup

while.body.lr.ph:
  %1 = load float*, float** %d, align 4
  %2 = load float*, float** %g, align 4
  br label %while.body

while.body:
  %i.012 = phi i32 [ 0, %while.body.lr.ph ], [ %inc, %if.end4 ]
  %rem = urem i32 %i.012, 10
  %tobool = icmp eq i32 %rem, 0
  br i1 %tobool, label %if.end4, label %if.then2

if.then2:
  %arrayidx = getelementptr inbounds float, float* %1, i32 %i.012
  %3 = load float, float* %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds float, float* %2, i32 %i.012
  %4 = load float, float* %arrayidx3, align 4
  %add = fadd float %3, %4
  store float %add, float* %arrayidx3, align 4
  br label %if.end4

if.end4:
  %inc = add nuw i32 %i.012, 1
  %cmp1 = icmp ult i32 %inc, %n
  br i1 %cmp1, label %while.body, label %cleanup.loopexit

cleanup.loopexit:
  br label %cleanup

cleanup:
  ret void
}
