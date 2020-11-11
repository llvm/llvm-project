; RUN: opt -mtriple=thumbv8.1m.main-none-none-eabi -hardware-loops %s -S -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-MAIN
; RUN: opt -mtriple=thumbv8.1m.main-none-none-eabi -mattr=+fullfp16 -hardware-loops %s -S -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-FP
; RUN: opt -mtriple=thumbv8.1m.main-none-none-eabi -mattr=+fp-armv8,+fullfp16 -hardware-loops %s -S -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-FP64
; RUN: opt -mtriple=thumbv8.1m.main-none-none-eabi -mattr=+mve -hardware-loops %s -S -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-MVE
; RUN: opt -mtriple=thumbv8.1m.main-none-none-eabi -mattr=+mve.fp -hardware-loops %s -S -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-MVEFP
; RUN: opt -mtriple=thumbv8.1m.main-none-none-eabi -hardware-loops -disable-arm-loloops=true %s -S -o - | FileCheck %s --check-prefix=DISABLED

; DISABLED-NOT: call i32 @llvm.loop.decrement

; CHECK-LABEL: skip_call
; CHECK-NOT: call i32 @llvm.start.loop.iterations
; CHECK-NOT: call i32 @llvm.loop.decrement

define i32 @skip_call(i32 %n) {
entry:
  %cmp6 = icmp eq i32 %n, 0
  br i1 %cmp6, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %i.08 = phi i32 [ %inc1, %while.body ], [ 0, %while.body.preheader ]
  %res.07 = phi i32 [ %add, %while.body ], [ 0, %while.body.preheader ]
  %call = tail call i32 bitcast (i32 (...)* @bar to i32 ()*)() #2
  %add = add nsw i32 %call, %res.07
  %inc1 = add nuw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc1, %n
  br i1 %exitcond, label %while.end.loopexit, label %while.body

while.end.loopexit:
  br label %while.end

while.end:
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %while.end.loopexit ]
  ret i32 %res.0.lcssa
}

; CHECK-LABEL: test_target_specific
; CHECK: [[X:%[^ ]+]] = call i32 @llvm.start.loop.iterations.i32(i32 50)
; CHECK: [[COUNT:%[^ ]+]] = phi i32 [ [[X]], %entry ], [ [[LOOP_DEC:%[^ ]+]], %loop ]
; CHECK: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[COUNT]], i32 1)
; CHECK: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK: br i1 [[CMP]], label %loop, label %exit

define i32 @test_target_specific(i32* %a, i32* %b) {
entry:
  br label %loop
loop:
  %acc = phi i32 [ 0, %entry ], [ %res, %loop ]
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %addr.a = getelementptr i32, i32* %a, i32 %count
  %addr.b = getelementptr i32, i32* %b, i32 %count
  %load.a = load i32, i32* %addr.a
  %load.b = load i32, i32* %addr.b
  %res = call i32 @llvm.arm.smlad(i32 %load.a, i32 %load.b, i32 %acc)
  %count.next = add nuw i32 %count, 2
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret i32 %res
}

; CHECK-LABEL: test_fabs_f16
; CHECK-MAIN-NOT: call i32 @llvm.start.loop.iterations
; CHECK-MVE-NOT:  call i32 @llvm.start.loop.iterations
; CHECK-FP:       call i32 @llvm.start.loop.iterations.i32(i32 100)
; CHECK-MVEFP:    call i32 @llvm.start.loop.iterations.i32(i32 100)
define void @test_fabs_f16(half* %a, half* %b) {
entry:
  br label %loop
loop:
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %addr.a = getelementptr half, half* %a, i32 %count
  %load.a = load half, half* %addr.a
  %abs = call half @llvm.fabs.f16(half %load.a)
  %addr.b = getelementptr half, half* %b, i32 %count
  store half %abs, half *%addr.b
  %count.next = add nuw i32 %count, 1
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

; CHECK-LABEL: test_fabs
; CHECK-MAIN-NOT: call i32 @llvm.start.loop.iterations
; CHECK-MVE-NOT:  call i32 @llvm.start.loop.iterations
; CHECK-FP:       call i32 @llvm.start.loop.iterations.i32(i32 100)
; CHECK-MVEFP:    call i32 @llvm.start.loop.iterations.i32(i32 100)

define float @test_fabs(float* %a) {
entry:
  br label %loop
loop:
  %acc = phi float [ 0.0, %entry ], [ %res, %loop ]
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %addr.a = getelementptr float, float* %a, i32 %count
  %load.a = load float, float* %addr.a
  %abs = call float @llvm.fabs.f32(float %load.a)
  %res = fadd float %abs, %acc
  %count.next = add nuw i32 %count, 1
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret float %res
}

; CHECK-LABEL: test_fabs_64
; CHECK-MAIN-NOT:   call i32 @llvm.start.loop.iterations
; CHECK-MVE-NOT:    call i32 @llvm.start.loop.iterations
; CHECK-FP-NOT:     call i32 @llvm.start.loop.iterations.i32(i32 100)
; CHECK-FP64:       call i32 @llvm.start.loop.iterations.i32(i32 100)
; CHECK-MVEFP-NOT:  call i32 @llvm.start.loop.iterations.i32(i32 100)
define void @test_fabs_64(double* %a, double* %b) {
entry:
  br label %loop
loop:
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %addr.a = getelementptr double, double* %a, i32 %count
  %load.a = load double, double* %addr.a
  %abs = call double @llvm.fabs.f64(double %load.a)
  %addr.b = getelementptr double, double* %b, i32 %count
  store double %abs, double *%addr.b
  %count.next = add nuw i32 %count, 1
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

; CHECK-LABEL: test_fabs_vec
; CHECK-MVE-NOT: call i32 @llvm.start.loop.iterations
; CHECK-MVEFP: [[X:%[^ ]+]] = call i32 @llvm.start.loop.iterations.i32(i32 100)
; CHECK-MVEFP: [[COUNT:%[^ ]+]] = phi i32 [ [[X]], %entry ], [ [[LOOP_DEC:%[^ ]+]], %loop ]
; CHECK-MVEFP: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[COUNT]], i32 1)
; CHECK-MVEFP: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK-MVEFP: br i1 [[CMP]], label %loop, label %exit
define <4 x float> @test_fabs_vec(<4 x float>* %a) {
entry:
  br label %loop
loop:
  %acc = phi <4 x float> [ zeroinitializer, %entry ], [ %res, %loop ]
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %addr.a = getelementptr <4 x float>, <4 x float>* %a, i32 %count
  %load.a = load <4 x float>, <4 x float>* %addr.a
  %abs = call <4 x float> @llvm.fabs.v4f32(<4 x float> %load.a)
  %res = fadd <4 x float> %abs, %acc
  %count.next = add nuw i32 %count, 1
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret <4 x float> %res
}

; CHECK-LABEL: test_log
; CHECK-NOT: call i32 @llvm.start.loop.iterations
; CHECK-NOT: llvm.loop.decrement
define float @test_log(float* %a) {
entry:
  br label %loop
loop:
  %acc = phi float [ 0.0, %entry ], [ %res, %loop ]
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %addr.a = getelementptr float, float* %a, i32 %count
  %load.a = load float, float* %addr.a
  %abs = call float @llvm.log.f32(float %load.a)
  %res = fadd float %abs, %acc
  %count.next = add nuw i32 %count, 1
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret float %res
}

; CHECK-LABEL: test_sqrt_16
; CHECK-MAIN-NOT: call i32 @llvm.start.loop.iterations
; CHECK-MVE-NOT:  call i32 @llvm.start.loop.iterations
; CHECK-FP:       call i32 @llvm.start.loop.iterations.i32(i32 100)
; CHECK-MVEFP:    call i32 @llvm.start.loop.iterations.i32(i32 100)
; CHECK-FP64:     call i32 @llvm.start.loop.iterations.i32(i32 100)
define void @test_sqrt_16(half* %a, half* %b) {
entry:
  br label %loop
loop:
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %addr.a = getelementptr half, half* %a, i32 %count
  %load.a = load half, half* %addr.a
  %sqrt = call half @llvm.sqrt.f16(half %load.a)
  %addr.b = getelementptr half, half* %b, i32 %count
  store half %sqrt, half *%addr.b
  %count.next = add nuw i32 %count, 1
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}
; CHECK-LABEL: test_sqrt
; CHECK-MAIN-NOT: call i32 @llvm.start.loop.iterations
; CHECK-MVE-NOT: call i32 @llvm.start.loop.iterations
; CHECK-FP: call i32 @llvm.start.loop.iterations
; CHECK-MVEFP: [[X:%[^ ]+]] = call i32 @llvm.start.loop.iterations.i32(i32 100)
; CHECK-MVEFP: [[COUNT:%[^ ]+]] = phi i32 [ [[X]], %entry ], [ [[LOOP_DEC:%[^ ]+]], %loop ]
; CHECK-MVEFP: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[COUNT]], i32 1)
; CHECK-MVEFP: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK-MVEFP: br i1 [[CMP]], label %loop, label %exit
define void @test_sqrt(float* %a, float* %b) {
entry:
  br label %loop
loop:
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %addr.a = getelementptr float, float* %a, i32 %count
  %load.a = load float, float* %addr.a
  %sqrt = call float @llvm.sqrt.f32(float %load.a)
  %addr.b = getelementptr float, float* %b, i32 %count
  store float %sqrt, float* %addr.b
  %count.next = add nuw i32 %count, 1
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

; CHECK-LABEL: test_sqrt_64
; CHECK-MAIN-NOT:   call i32 @llvm.start.loop.iterations
; CHECK-MVE-NOT:    call i32 @llvm.start.loop.iterations
; CHECK-FP-NOT:     call i32 @llvm.start.loop.iterations.i32(i32 100)
; CHECK-MVEFP-NOT:  call i32 @llvm.start.loop.iterations.i32(i32 100)
; CHECK-FP64:       call i32 @llvm.start.loop.iterations.i32(i32 100)
define void @test_sqrt_64(double* %a, double* %b) {
entry:
  br label %loop
loop:
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %addr.a = getelementptr double, double* %a, i32 %count
  %load.a = load double, double* %addr.a
  %sqrt = call double @llvm.sqrt.f64(double %load.a)
  %addr.b = getelementptr double, double* %b, i32 %count
  store double %sqrt, double *%addr.b
  %count.next = add nuw i32 %count, 1
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

; CHECK-LABEL: test_sqrt_vec
; CHECK-MAIN-NOT: call i32 @llvm.start.loop.iterations
; CHECK-MVE-NOT:  call i32 @llvm.start.loop.iterations
; CHECK-FP:       call i32 @llvm.start.loop.iterations.i32(i32 100)
; CHECK-MVEFP:    call i32 @llvm.start.loop.iterations.i32(i32 100)
define void @test_sqrt_vec(<4 x float>* %a, <4 x float>* %b) {
entry:
  br label %loop
loop:
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %addr.a = getelementptr <4 x float>, <4 x float>* %a, i32 %count
  %load.a = load <4 x float>, <4 x float>* %addr.a
  %sqrt = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %load.a)
  %addr.b = getelementptr <4 x float>, <4 x float>* %b, i32 %count
  store <4 x float> %sqrt, <4 x float>* %addr.b
  %count.next = add nuw i32 %count, 1
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

; CHECK-LABEL: test_overflow
; CHECK: call i32 @llvm.start.loop.iterations
define i32 @test_overflow(i32* %a, i32* %b) {
entry:
  br label %loop
loop:
  %acc = phi i32 [ 0, %entry ], [ %res, %loop ]
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %addr.a = getelementptr i32, i32* %a, i32 %count
  %addr.b = getelementptr i32, i32* %b, i32 %count
  %load.a = load i32, i32* %addr.a
  %load.b = load i32, i32* %addr.b
  %sadd = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %load.a, i32 %load.b)
  %res = extractvalue {i32, i1} %sadd, 0
  %count.next = add nuw i32 %count, 1
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret i32 %res
}

; TODO: We should be able to generate a qadd/sub
; CHECK-LABEL: test_sat
; CHECK: call i32 @llvm.start.loop.iterations.i32(i32 100)
define i32 @test_sat(i32* %a, i32* %b) {
entry:
  br label %loop
loop:
  %acc = phi i32 [ 0, %entry ], [ %res, %loop ]
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %addr.a = getelementptr i32, i32* %a, i32 %count
  %addr.b = getelementptr i32, i32* %b, i32 %count
  %load.a = load i32, i32* %addr.a
  %load.b = load i32, i32* %addr.b
  %res = call i32 @llvm.sadd.sat.i32(i32 %load.a, i32 %load.b)
  %count.next = add nuw i32 %count, 1
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret i32 %res
}

; CHECK-LABEL: test_masked_i32
; CHECK-NOT: call i32 @llvm.start.loop.iterations
; CHECK-MVEFP: call i32 @llvm.start.loop.iterations
; CHECK-MVE: [[X:%[^ ]+]] = call i32 @llvm.start.loop.iterations.i32(i32 100)
; CHECK-MVE: [[COUNT:%[^ ]+]] = phi i32 [ [[X]], %entry ], [ [[LOOP_DEC:%[^ ]+]], %loop ]
; CHECK-MVE: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[COUNT]], i32 1)
; CHECK-MVE: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK-MVE: br i1 [[CMP]], label %loop, label %exit
define arm_aapcs_vfpcc void @test_masked_i32(<4 x i1> %mask, <4 x i32>* %a, <4 x i32>* %b, <4 x i32>* %c, <4 x i32> %passthru) {
entry:
  br label %loop
loop:
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %addr.a = getelementptr <4 x i32>, <4 x i32>* %a, i32 %count
  %addr.b = getelementptr <4 x i32>, <4 x i32>* %b, i32 %count
  %addr.c = getelementptr <4 x i32>, <4 x i32>* %c, i32 %count
  %load.a = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %addr.a, i32 4, <4 x i1> %mask, <4 x i32> %passthru)
  %load.b = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %addr.b, i32 4, <4 x i1> %mask, <4 x i32> %passthru)
  %res = add <4 x i32> %load.a, %load.b
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %res, <4 x i32>* %addr.c, i32 4, <4 x i1> %mask)
  %count.next = add nuw i32 %count, 1
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

; CHECK-LABEL: test_masked_f32
; CHECK-NOT: call i32 @llvm.start.loop.iterations
; CHECK-MVEFP: call i32 @llvm.start.loop.iterations
; CHECK-MVE: [[X:%[^ ]+]] = call i32 @llvm.start.loop.iterations.i32(i32 100)
; CHECK-MVE: [[COUNT:%[^ ]+]] = phi i32 [ [[X]], %entry ], [ [[LOOP_DEC:%[^ ]+]], %loop ]
; CHECK-MVE: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[COUNT]], i32 1)
; CHECK-MVE: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK-MVE: br i1 [[CMP]], label %loop, label %exit
define arm_aapcs_vfpcc void @test_masked_f32(<4 x i1> %mask, <4 x float>* %a, <4 x float>* %b, <4 x float>* %c, <4 x float> %passthru) {
entry:
  br label %loop
loop:
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %addr.a = getelementptr <4 x float>, <4 x float>* %a, i32 %count
  %addr.b = getelementptr <4 x float>, <4 x float>* %b, i32 %count
  %addr.c = getelementptr <4 x float>, <4 x float>* %c, i32 %count
  %load.a = call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %addr.a, i32 4, <4 x i1> %mask, <4 x float> %passthru)
  %load.b = call <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>* %addr.b, i32 4, <4 x i1> %mask, <4 x float> %passthru)
  %res = fadd <4 x float> %load.a, %load.b
  call void @llvm.masked.store.v4f32.p0v4f32(<4 x float> %res, <4 x float>* %addr.c, i32 4, <4 x i1> %mask)
  %count.next = add nuw i32 %count, 1
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

; CHECK-LABEL: test_gather_scatter
; CHECK-NOT: call i32 @llvm.start.loop.iterations
; CHECK-MVEFP: call i32 @llvm.start.loop.iterations
; CHECK-MVE: [[X:%[^ ]+]] = call i32 @llvm.start.loop.iterations.i32(i32 100)
; CHECK-MVE: [[COUNT:%[^ ]+]] = phi i32 [ [[X]], %entry ], [ [[LOOP_DEC:%[^ ]+]], %loop ]
; CHECK-MVE: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32(i32 [[COUNT]], i32 1)
; CHECK-MVE: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK-MVE: br i1 [[CMP]], label %loop, label %exit
define arm_aapcs_vfpcc void @test_gather_scatter(<4 x i1> %mask, <4 x float*> %a, <4 x float*> %b, <4 x float*> %c, <4 x float> %passthru) {
entry:
  br label %loop
loop:
  %count = phi i32 [ 0, %entry ], [ %count.next, %loop ]
  %load.a = call <4 x float> @llvm.masked.gather.v4f32.p0v4f32(<4 x float*> %a, i32 4, <4 x i1> %mask, <4 x float> %passthru)
  %load.b = call <4 x float> @llvm.masked.gather.v4f32.p0v4f32(<4 x float*> %b, i32 4, <4 x i1> %mask, <4 x float> %passthru)
  %res = fadd <4 x float> %load.a, %load.b
  call void @llvm.masked.scatter.v4f32.p0v4f32(<4 x float> %res, <4 x float*> %c, i32 4, <4 x i1> %mask)
  %count.next = add nuw i32 %count, 1
  %cmp = icmp ne i32 %count.next, 100
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

declare i32 @bar(...) local_unnamed_addr #1
declare i32 @llvm.arm.smlad(i32, i32, i32)
declare half @llvm.fabs.f16(half)
declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare float @llvm.log.f32(float)
declare <4 x float> @llvm.fabs.v4f32(<4 x float>)
declare half @llvm.sqrt.f16(half)
declare float @llvm.sqrt.f32(float)
declare double @llvm.sqrt.f64(double)
declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)
declare i32 @llvm.sadd.sat.i32(i32, i32)
declare {i32, i1} @llvm.sadd.with.overflow.i32(i32, i32)
declare <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>*, i32, <4 x i1>, <4 x i32>)
declare void @llvm.masked.store.v4i32.p0v4i32(<4 x i32>, <4 x i32>*, i32, <4 x i1>)
declare <4 x float> @llvm.masked.load.v4f32.p0v4f32(<4 x float>*, i32, <4 x i1>, <4 x float>)
declare void @llvm.masked.store.v4f32.p0v4f32(<4 x float>, <4 x float>*, i32, <4 x i1>)
declare <4 x float> @llvm.masked.gather.v4f32.p0v4f32(<4 x float*>, i32, <4 x i1>, <4 x float>)
declare void @llvm.masked.scatter.v4f32.p0v4f32(<4 x float>, <4 x float*>, i32, <4 x i1>)
