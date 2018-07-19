; RUN: llc < %s -mtriple=thumbv7-elf -mattr=+neon -arm-atomic-cfg-tidy=0 | FileCheck %s
; PR4789

%bar = type { float, float, float }
%baz = type { i32, [16 x %bar], [16 x float], [16 x i32], i8 }
%foo = type { <4 x float> }
%quux = type { i32 (...)**, %baz*, i32 }
%quuz = type { %quux, i32, %bar, [128 x i8], [16 x %foo], %foo, %foo, %foo }

declare <4 x float> @llvm.arm.neon.vld1.v4f32.p0i8(i8*, i32) nounwind readonly

define void @aaa(%quuz* %this, i8* %block) {
; CHECK-LABEL: aaa:
; CHECK: bfc r4, #0, #4
; CHECK: vst1.64 {{.*}}[{{.*}}:128]
; CHECK: vld1.64 {{.*}}[{{.*}}:128]
entry:
  %aligned_vec = alloca <4 x float>, align 16
  %"alloca point" = bitcast i32 0 to i32
  %vecptr = bitcast <4 x float>* %aligned_vec to i8*
  %0 = call <4 x float> @llvm.arm.neon.vld1.v4f32.p0i8(i8* %vecptr, i32 1) nounwind 
  store float 6.300000e+01, float* undef, align 4
  %1 = call <4 x float> @llvm.arm.neon.vld1.v4f32.p0i8(i8* undef, i32 1) nounwind ; <<4 x float>> [#uses=1]
  store float 0.000000e+00, float* undef, align 4
  %2 = call <4 x float> @llvm.arm.neon.vld1.v4f32.p0i8(i8* undef, i32 1) nounwind ; <<4 x float>> [#uses=1]
  %ld3 = call <4 x float> @llvm.arm.neon.vld1.v4f32.p0i8(i8* undef, i32 1) nounwind
  store float 0.000000e+00, float* undef, align 4
  %ld4 = call <4 x float> @llvm.arm.neon.vld1.v4f32.p0i8(i8* undef, i32 1) nounwind
  store float 0.000000e+00, float* undef, align 4
  %ld5 = call <4 x float> @llvm.arm.neon.vld1.v4f32.p0i8(i8* undef, i32 1) nounwind
  store float 0.000000e+00, float* undef, align 4
  %ld6 = call <4 x float> @llvm.arm.neon.vld1.v4f32.p0i8(i8* undef, i32 1) nounwind
  store float 0.000000e+00, float* undef, align 4
  %ld7 = call <4 x float> @llvm.arm.neon.vld1.v4f32.p0i8(i8* undef, i32 1) nounwind
  store float 0.000000e+00, float* undef, align 4
  %ld8 = call <4 x float> @llvm.arm.neon.vld1.v4f32.p0i8(i8* undef, i32 1) nounwind
  store float 0.000000e+00, float* undef, align 4
  %ld9 = call <4 x float> @llvm.arm.neon.vld1.v4f32.p0i8(i8* undef, i32 1) nounwind
  store float 0.000000e+00, float* undef, align 4
  %ld10 = call <4 x float> @llvm.arm.neon.vld1.v4f32.p0i8(i8* undef, i32 1) nounwind
  store float 0.000000e+00, float* undef, align 4
  %ld11 = call <4 x float> @llvm.arm.neon.vld1.v4f32.p0i8(i8* undef, i32 1) nounwind
  store float 0.000000e+00, float* undef, align 4
  %ld12 = call <4 x float> @llvm.arm.neon.vld1.v4f32.p0i8(i8* undef, i32 1) nounwind
  store float 0.000000e+00, float* undef, align 4
  %val173 = load <4 x float>, <4 x float>* undef               ; <<4 x float>> [#uses=1]
  br label %bb4

bb4:                                              ; preds = %bb193, %entry
  %besterror.0.2264 = phi <4 x float> [ undef, %entry ], [ %besterror.0.0, %bb193 ] ; <<4 x float>> [#uses=2]
  %part0.0.0261 = phi <4 x float> [ zeroinitializer, %entry ], [ %23, %bb193 ] ; <<4 x float>> [#uses=2]
  %3 = fmul <4 x float> zeroinitializer, %0       ; <<4 x float>> [#uses=2]
  %4 = fadd <4 x float> %3, %part0.0.0261         ; <<4 x float>> [#uses=1]
  %5 = shufflevector <4 x float> %3, <4 x float> undef, <2 x i32> <i32 2, i32 3> ; <<2 x float>> [#uses=1]
  %6 = shufflevector <2 x float> %5, <2 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x float>> [#uses=1]
  %7 = fmul <4 x float> %1, undef                 ; <<4 x float>> [#uses=1]
  %8 = fadd <4 x float> %7, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01> ; <<4 x float>> [#uses=1]
  %9 = fptosi <4 x float> %8 to <4 x i32>         ; <<4 x i32>> [#uses=1]
  %10 = sitofp <4 x i32> %9 to <4 x float>        ; <<4 x float>> [#uses=1]
  %11 = fmul <4 x float> %10, %2                  ; <<4 x float>> [#uses=1]
  %12 = fmul <4 x float> undef, %6                ; <<4 x float>> [#uses=1]
  %13 = fmul <4 x float> %11, %4                  ; <<4 x float>> [#uses=1]
  %14 = fsub <4 x float> %12, %13                 ; <<4 x float>> [#uses=1]
  %15 = fsub <4 x float> %14, undef               ; <<4 x float>> [#uses=1]
  %16 = fmul <4 x float> %15, <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00> ; <<4 x float>> [#uses=1]
  %17 = fadd <4 x float> %16, undef               ; <<4 x float>> [#uses=1]
  %18 = fmul <4 x float> %17, %val173             ; <<4 x float>> [#uses=1]
  %19 = shufflevector <4 x float> %18, <4 x float> undef, <2 x i32> <i32 2, i32 3> ; <<2 x float>> [#uses=1]
  %20 = shufflevector <2 x float> %19, <2 x float> undef, <4 x i32> zeroinitializer ; <<4 x float>> [#uses=1]
  %tmp1 = fadd <4 x float> %20, %ld3
  %tmp2 = fadd <4 x float> %tmp1, %ld4
  %tmp3 = fadd <4 x float> %tmp2, %ld5
  %tmp4 = fadd <4 x float> %tmp3, %ld6
  %tmp5 = fadd <4 x float> %tmp4, %ld7
  %tmp6 = fadd <4 x float> %tmp5, %ld8
  %tmp7 = fadd <4 x float> %tmp6, %ld9
  %tmp8 = fadd <4 x float> %tmp7, %ld10
  %tmp9 = fadd <4 x float> %tmp8, %ld11
  %21 = fadd <4 x float> %tmp9, %ld12
  %22 = fcmp ogt <4 x float> %besterror.0.2264, %21 ; <<4 x i1>> [#uses=0]
  %tmp = extractelement <4 x i1> %22, i32 0
  br i1 %tmp, label %bb193, label %bb186

bb186:                                            ; preds = %bb4
  br label %bb193

bb193:                                            ; preds = %bb186, %bb4
  %besterror.0.0 = phi <4 x float> [ %21, %bb186 ], [ %besterror.0.2264, %bb4 ] ; <<4 x float>> [#uses=1]
  %23 = fadd <4 x float> %part0.0.0261, zeroinitializer ; <<4 x float>> [#uses=1]
  br label %bb4
}
