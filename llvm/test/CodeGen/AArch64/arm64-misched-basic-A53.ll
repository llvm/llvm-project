; REQUIRES: asserts
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=cortex-a53 -pre-RA-sched=source -enable-misched -verify-misched -debug-only=machine-scheduler -disable-machine-dce -o - 2>&1 > /dev/null | FileCheck %s
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=cortex-a53 -pre-RA-sched=source -enable-misched -verify-misched -debug-only=machine-scheduler -disable-machine-dce -o - -misched-limit=2 2>&1 > /dev/null | FileCheck %s
;
; The Cortex-A53 machine model will cause the MADD instruction to be scheduled
; much higher than the ADD instructions in order to hide latency. When not
; specifying a subtarget, the MADD will remain near the end of the block.
;
; CHECK: ********** MI Scheduling **********
; CHECK: main
; CHECK: *** Final schedule for BB#2 ***
; CHECK: MADDWrrr
; CHECK: ADDWri
; CHECK: ********** INTERVALS **********
@main.x = private unnamed_addr constant [8 x i32] [i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1], align 4
@main.y = private unnamed_addr constant [8 x i32] [i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2], align 4

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %x = alloca [8 x i32], align 4
  %y = alloca [8 x i32], align 4
  %i = alloca i32, align 4
  %xx = alloca i32, align 4
  %yy = alloca i32, align 4
  store i32 0, i32* %retval
  %0 = bitcast [8 x i32]* %x to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* bitcast ([8 x i32]* @main.x to i8*), i64 32, i32 4, i1 false)
  %1 = bitcast [8 x i32]* %y to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* bitcast ([8 x i32]* @main.y to i8*), i64 32, i32 4, i1 false)
  store i32 0, i32* %xx, align 4
  store i32 0, i32* %yy, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %2, 8
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load i32, i32* %i, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds [8 x i32], [8 x i32]* %x, i32 0, i64 %idxprom
  %4 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %4, 1
  store i32 %add, i32* %xx, align 4
  %5 = load i32, i32* %xx, align 4
  %add1 = add nsw i32 %5, 12
  store i32 %add1, i32* %xx, align 4
  %6 = load i32, i32* %xx, align 4
  %add2 = add nsw i32 %6, 23
  store i32 %add2, i32* %xx, align 4
  %7 = load i32, i32* %xx, align 4
  %add3 = add nsw i32 %7, 34
  store i32 %add3, i32* %xx, align 4
  %8 = load i32, i32* %i, align 4
  %idxprom4 = sext i32 %8 to i64
  %arrayidx5 = getelementptr inbounds [8 x i32], [8 x i32]* %y, i32 0, i64 %idxprom4
  %9 = load i32, i32* %arrayidx5, align 4
  %10 = load i32, i32* %yy, align 4
  %mul = mul nsw i32 %10, %9
  store i32 %mul, i32* %yy, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %11 = load i32, i32* %i, align 4
  %inc = add nsw i32 %11, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %12 = load i32, i32* %xx, align 4
  %13 = load i32, i32* %yy, align 4
  %add6 = add nsw i32 %12, %13
  ret i32 %add6
}


; The Cortex-A53 machine model will cause the FDIVvvv_42 to be raised to
; hide latency. Whereas normally there would only be a single FADDvvv_4s
; after it, this test checks to make sure there are more than one.
;
; CHECK: ********** MI Scheduling **********
; CHECK: neon4xfloat:BB#0
; CHECK: *** Final schedule for BB#0 ***
; CHECK: FDIVv4f32
; CHECK: FADDv4f32
; CHECK: FADDv4f32
; CHECK: ********** INTERVALS **********
define <4 x float> @neon4xfloat(<4 x float> %A, <4 x float> %B) {
        %tmp1 = fadd <4 x float> %A, %B;
        %tmp2 = fadd <4 x float> %A, %tmp1;
        %tmp3 = fadd <4 x float> %A, %tmp2;
        %tmp4 = fadd <4 x float> %A, %tmp3;
        %tmp5 = fadd <4 x float> %A, %tmp4;
        %tmp6 = fadd <4 x float> %A, %tmp5;
        %tmp7 = fadd <4 x float> %A, %tmp6;
        %tmp8 = fadd <4 x float> %A, %tmp7;
        %tmp9 = fdiv <4 x float> %A, %B;
        %tmp10 = fadd <4 x float> %tmp8, %tmp9;

        ret <4 x float> %tmp10
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }


; Regression Test for PR19761
;   [ARM64] Cortex-a53 schedule mode can't handle NEON post-increment load
;
; Nothing explicit to check other than llc not crashing.
define { <16 x i8>, <16 x i8> } @test_v16i8_post_imm_ld2(i8* %A, i8** %ptr) {
  %ld2 = tail call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld2.v16i8.p0i8(i8* %A)
  %tmp = getelementptr i8, i8* %A, i32 32
  store i8* %tmp, i8** %ptr
  ret { <16 x i8>, <16 x i8> } %ld2
}

declare { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld2.v16i8.p0i8(i8*)

; Regression Test for PR20057.
;
; Cortex-A53 machine model stalls on A53UnitFPMDS contention. Instructions that
; are otherwise ready are jammed in the pending queue.
; CHECK: ********** MI Scheduling **********
; CHECK: testResourceConflict
; CHECK: *** Final schedule for BB#0 ***
; CHECK: BRK
; CHECK: ********** INTERVALS **********
define void @testResourceConflict(float* %ptr) {
entry:
  %add1 = fadd float undef, undef
  %mul2 = fmul float undef, undef
  %add3 = fadd float %mul2, undef
  %mul4 = fmul float undef, %add3
  %add5 = fadd float %mul4, undef
  %sub6 = fsub float 0.000000e+00, undef
  %sub7 = fsub float %add5, undef
  %div8 = fdiv float 1.000000e+00, undef
  %mul9 = fmul float %div8, %sub7
  %mul14 = fmul float %sub6, %div8
  %mul10 = fsub float -0.000000e+00, %mul14
  %mul15 = fmul float undef, %div8
  %mul11 = fsub float -0.000000e+00, %mul15
  %mul12 = fmul float 0.000000e+00, %div8
  %mul13 = fmul float %add1, %mul9
  %mul21 = fmul float %add5, %mul11
  %add22 = fadd float %mul13, %mul21
  store float %add22, float* %ptr, align 4
  %mul28 = fmul float %add1, %mul10
  %mul33 = fmul float %add5, %mul12
  %add34 = fadd float %mul33, %mul28
  store float %add34, float* %ptr, align 4
  %mul240 = fmul float undef, %mul9
  %add246 = fadd float %mul240, undef
  store float %add246, float* %ptr, align 4
  %mul52 = fmul float undef, %mul10
  %mul57 = fmul float undef, %mul12
  %add58 = fadd float %mul57, %mul52
  store float %add58, float* %ptr, align 4
  %mul27 = fmul float 0.000000e+00, %mul9
  %mul81 = fmul float undef, %mul10
  %add82 = fadd float %mul27, %mul81
  store float %add82, float* %ptr, align 4
  call void @llvm.trap()
  unreachable
}

declare void @llvm.trap()

; Regression test for PR20057: "permanent hazard"'
; Resource contention on LDST.
; CHECK: ********** MI Scheduling **********
; CHECK: testLdStConflict
; CHECK: *** Final schedule for BB#1 ***
; CHECK: LD4Fourv2d
; CHECK: STRQui
; CHECK: ********** INTERVALS **********
define void @testLdStConflict(<2 x i64> %v) {
entry:
  br label %loop

loop:
  %0 = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld4.v2i64.p0i8(i8* null)
  %ptr = bitcast i8* undef to <2 x i64>*
  store <2 x i64> %v, <2 x i64>* %ptr, align 4
  %ptr1 = bitcast i8* undef to <2 x i64>*
  store <2 x i64> %v, <2 x i64>* %ptr1, align 4
  %ptr2 = bitcast i8* undef to <2 x i64>*
  store <2 x i64> %v, <2 x i64>* %ptr2, align 4
  %ptr3 = bitcast i8* undef to <2 x i64>*
  store <2 x i64> %v, <2 x i64>* %ptr3, align 4
  %ptr4 = bitcast i8* undef to <2 x i64>*
  store <2 x i64> %v, <2 x i64>* %ptr4, align 4
  br label %loop
}

declare { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld4.v2i64.p0i8(i8*)
