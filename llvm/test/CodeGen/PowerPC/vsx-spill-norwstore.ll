; RUN: llc -mcpu=pwr7 -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@.str1 = external unnamed_addr constant [5 x i8], align 1
@.str10 = external unnamed_addr constant [9 x i8], align 1
@.v2f64 = external unnamed_addr constant <2 x double>, align 16

; Function Attrs: nounwind
define void @main() #0 {
; CHECK-LABEL: @main
; Make sure that the stxvd2x passes -verify-machineinstrs
; CHECK: stxvd2x

entry:
  %val = load <2 x double>, ptr @.v2f64, align 16
  %0 = tail call <8 x i16> @llvm.ppc.altivec.vupkhsb(<16 x i8> <i8 0, i8 -1, i8 -1, i8 0, i8 0, i8 0, i8 -1, i8 0, i8 -1, i8 0, i8 0, i8 -1, i8 -1, i8 -1, i8 0, i8 -1>) #0
  %1 = tail call <8 x i16> @llvm.ppc.altivec.vupklsb(<16 x i8> <i8 0, i8 -1, i8 -1, i8 0, i8 0, i8 0, i8 -1, i8 0, i8 -1, i8 0, i8 0, i8 -1, i8 -1, i8 -1, i8 0, i8 -1>) #0
  br i1 false, label %if.then.i68.i, label %check.exit69.i

if.then.i68.i:                                    ; preds = %entry
  unreachable

check.exit69.i:                                   ; preds = %entry
  br i1 undef, label %if.then.i63.i, label %check.exit64.i

if.then.i63.i:                                    ; preds = %check.exit69.i
  tail call void (ptr, ...) @printf(ptr @.str10, ptr @.str1, <2 x double> %val) #0
  br label %check.exit64.i

check.exit64.i:                                   ; preds = %if.then.i63.i, %check.exit69.i
  %2 = tail call i32 @llvm.ppc.altivec.vcmpequh.p(i32 2, <8 x i16> %0, <8 x i16> <i16 0, i16 -1, i16 -1, i16 0, i16 0, i16 0, i16 -1, i16 0>) #0
  %tobool.i55.i = icmp eq i32 %2, 0
  br i1 %tobool.i55.i, label %if.then.i58.i, label %check.exit59.i

if.then.i58.i:                                    ; preds = %check.exit64.i
  unreachable

check.exit59.i:                                   ; preds = %check.exit64.i
  %3 = tail call i32 @llvm.ppc.altivec.vcmpequh.p(i32 2, <8 x i16> %1, <8 x i16> <i16 -1, i16 0, i16 0, i16 -1, i16 -1, i16 -1, i16 0, i16 -1>) #0
  %tobool.i50.i = icmp eq i32 %3, 0
  br i1 %tobool.i50.i, label %if.then.i53.i, label %check.exit54.i

if.then.i53.i:                                    ; preds = %check.exit59.i
  unreachable

check.exit54.i:                                   ; preds = %check.exit59.i
  unreachable
}

; Function Attrs: nounwind readnone
declare <8 x i16> @llvm.ppc.altivec.vupkhsb(<16 x i8>) #1

; Function Attrs: nounwind readnone
declare <8 x i16> @llvm.ppc.altivec.vupklsb(<16 x i8>) #1

; Function Attrs: nounwind
declare void @printf(ptr nocapture readonly, ...) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.ppc.altivec.vcmpequh.p(i32, <8 x i16>, <8 x i16>) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

