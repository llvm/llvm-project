; RUN: llc < %s -O3 -relocation-model=pic -mattr=+thumb2 -mcpu=cortex-a8 -disable-branch-fold | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

; This is a case where the coalescer was too eager. These two copies were
; considered equivalent and coalescable:
;
; 140 %reg1038:dsub_0<def> = VMOVD %reg1047:dsub_0, pred:14, pred:%reg0
; 148 %reg1038:dsub_1<def> = VMOVD %reg1047:dsub_0, pred:14, pred:%reg0
;
; Only one can be coalesced.

@.str = private constant [7 x i8] c"%g %g\0A\00", align 4 ; <[7 x i8]*> [#uses=1]

define i32 @main(i32 %argc, i8** nocapture %Argv) nounwind {
entry:
  %0 = icmp eq i32 %argc, 2123                    ; <i1> [#uses=1]
  %U.0 = select i1 %0, double 3.282190e+01, double 8.731834e+02 ; <double> [#uses=2]
  %1 = icmp eq i32 %argc, 5123                    ; <i1> [#uses=1]
  %V.0.ph = select i1 %1, double 7.779980e+01, double 0x409CCB9C779A6B51 ; <double> [#uses=1]
  %2 = insertelement <2 x double> undef, double %U.0, i32 0 ; <<2 x double>> [#uses=2]
  %3 = insertelement <2 x double> %2, double %U.0, i32 1 ; <<2 x double>> [#uses=2]
  %4 = insertelement <2 x double> %2, double %V.0.ph, i32 1 ; <<2 x double>> [#uses=2]
; Constant pool load followed by add.
; Then clobber the loaded register, not the sum.
; CHECK: vldr [[LDR:d.*]],
; CHECK: LPC0_0:
; CHECK: vadd.f64 [[ADD:d.*]], [[LDR]], [[LDR]]
; CHECK-NOT: vmov.f64 [[ADD]]
  %5 = fadd <2 x double> %3, %3                   ; <<2 x double>> [#uses=2]
  %6 = fadd <2 x double> %4, %4                   ; <<2 x double>> [#uses=2]
  %tmp7 = extractelement <2 x double> %5, i32 0   ; <double> [#uses=1]
  %tmp5 = extractelement <2 x double> %5, i32 1   ; <double> [#uses=1]
; CHECK: printf
  %7 = tail call  i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([7 x i8]* @.str, i32 0, i32 0), double %tmp7, double %tmp5) nounwind ; <i32> [#uses=0]
  %tmp3 = extractelement <2 x double> %6, i32 0   ; <double> [#uses=1]
  %tmp1 = extractelement <2 x double> %6, i32 1   ; <double> [#uses=1]
  %8 = tail call  i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([7 x i8]* @.str, i32 0, i32 0), double %tmp3, double %tmp1) nounwind ; <i32> [#uses=0]
  ret i32 0
}

declare i32 @printf(i8* nocapture, ...) nounwind
