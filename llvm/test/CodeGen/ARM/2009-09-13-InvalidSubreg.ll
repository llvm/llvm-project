; RUN: llc -mattr=+neon < %s
; PR4965
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv7-eabi"

%struct.fr = type { [6 x %struct.pl] }
%struct.obb = type { %"struct.m4", %"struct.p3" }
%struct.pl = type { %"struct.p3" }
%"struct.m4" = type { %"struct.p3", %"struct.p3", %"struct.p3", %"struct.p3" }
%"struct.p3" = type { <4 x float> }

declare <2 x float> @llvm.arm.neon.vpadd.v2f32(<2 x float>, <2 x float>) nounwind readnone

define arm_aapcs_vfpcc i8 @foo(%struct.fr* nocapture %this, %struct.obb* %box) nounwind {
entry:
  %val.i.i = load <4 x float>* undef              ; <<4 x float>> [#uses=1]
  %val2.i.i = load <4 x float>* null              ; <<4 x float>> [#uses=1]
  %elt3.i.i = getelementptr inbounds %struct.obb* %box, i32 0, i32 0, i32 2, i32 0 ; <<4 x float>*> [#uses=1]
  %val4.i.i = load <4 x float>* %elt3.i.i         ; <<4 x float>> [#uses=1]
  %0 = shufflevector <2 x float> undef, <2 x float> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3> ; <<4 x float>> [#uses=1]
  %1 = fadd <4 x float> undef, zeroinitializer    ; <<4 x float>> [#uses=1]
  br label %bb33

bb:                                               ; preds = %bb33
  %2 = fmul <4 x float> %val.i.i, undef           ; <<4 x float>> [#uses=1]
  %3 = fmul <4 x float> %val2.i.i, undef          ; <<4 x float>> [#uses=1]
  %4 = fadd <4 x float> %3, %2                    ; <<4 x float>> [#uses=1]
  %5 = fmul <4 x float> %val4.i.i, undef          ; <<4 x float>> [#uses=1]
  %6 = fadd <4 x float> %5, %4                    ; <<4 x float>> [#uses=1]
  %7 = bitcast <4 x float> %6 to <4 x i32>        ; <<4 x i32>> [#uses=1]
  %8 = and <4 x i32> %7, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648> ; <<4 x i32>> [#uses=1]
  %9 = or <4 x i32> %8, undef                     ; <<4 x i32>> [#uses=1]
  %10 = bitcast <4 x i32> %9 to <4 x float>       ; <<4 x float>> [#uses=1]
  %11 = shufflevector <4 x float> %10, <4 x float> undef, <2 x i32> <i32 0, i32 1> ; <<2 x float>> [#uses=1]
  %12 = shufflevector <2 x float> %11, <2 x float> undef, <4 x i32> zeroinitializer ; <<4 x float>> [#uses=1]
  %13 = fmul <4 x float> undef, %12               ; <<4 x float>> [#uses=1]
  %14 = fmul <4 x float> %0, undef                ; <<4 x float>> [#uses=1]
  %15 = fadd <4 x float> %14, %13                 ; <<4 x float>> [#uses=1]
  %16 = fadd <4 x float> undef, %15               ; <<4 x float>> [#uses=1]
  %17 = fadd <4 x float> %1, %16                  ; <<4 x float>> [#uses=1]
  %18 = fmul <4 x float> zeroinitializer, %17     ; <<4 x float>> [#uses=1]
  %19 = insertelement <4 x float> %18, float 0.000000e+00, i32 3 ; <<4 x float>> [#uses=2]
  %20 = shufflevector <4 x float> %19, <4 x float> undef, <2 x i32> <i32 0, i32 1> ; <<2 x float>> [#uses=1]
  %21 = shufflevector <4 x float> %19, <4 x float> undef, <2 x i32> <i32 2, i32 3> ; <<2 x float>> [#uses=1]
  %22 = tail call <2 x float> @llvm.arm.neon.vpadd.v2f32(<2 x float> %20, <2 x float> %21) nounwind ; <<2 x float>> [#uses=2]
  %23 = tail call <2 x float> @llvm.arm.neon.vpadd.v2f32(<2 x float> %22, <2 x float> %22) nounwind ; <<2 x float>> [#uses=2]
  %24 = shufflevector <2 x float> %23, <2 x float> %23, <4 x i32> zeroinitializer ; <<4 x float>> [#uses=1]
  %25 = fadd <4 x float> %24, zeroinitializer     ; <<4 x float>> [#uses=1]
  %tmp46 = extractelement <4 x float> %25, i32 0  ; <float> [#uses=1]
  %26 = fcmp olt float %tmp46, 0.000000e+00       ; <i1> [#uses=1]
  br i1 %26, label %bb41, label %bb33

bb33:                                             ; preds = %bb, %entry
  br i1 undef, label %bb34, label %bb

bb34:                                             ; preds = %bb33
  ret i8 undef

bb41:                                             ; preds = %bb
  ret i8 1
}
