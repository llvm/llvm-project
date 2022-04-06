; REQUIRES: arm-registered-target
; REQUIRES: asserts
; REQUIRES: abi_breaking_checks
; RUN: llc -o /dev/null %s -debug-only=legalize-types 2>&1 | FileCheck %s

; This test check that when v4f64 gets broken down to two v2f64 it maintains
; the "nnan" flags.

; CHECK: Legalizing node: [[VFOUR:t.*]]: v4f64 = BUILD_VECTOR
; CHECK-NEXT: Analyzing result type: v4f64
; CHECK-NEXT: Split node result: [[VFOUR]]: v4f64 = BUILD_VECTOR

; CHECK: Legalizing node: [[VTWOA:t.*]]: v2f64 = BUILD_VECTOR
; CHECK: Legally typed node: [[VTWOA]]: v2f64 = BUILD_VECTOR
; CHECK: Legalizing node: [[VTWOB:t.*]]: v2f64 = BUILD_VECTOR
; CHECK: Legally typed node: [[VTWOB]]: v2f64 = BUILD_VECTOR
; CHECK: Legalizing node: t34: v2f64 = fmaxnum nnan reassoc [[VTWOB]], [[VTWOA]]

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"


; Function Attrs: norecurse nounwind
define fastcc double @test(double %a0, double %a1, double %a2, double %a3) unnamed_addr #1 {
entry:
 %0 = insertelement <4 x double> undef, double %a0, i32 0
 %1 = insertelement <4 x double> %0, double %a1, i32 1
 %2 = insertelement <4 x double> %1, double %a2, i32 2
 %3 = insertelement <4 x double> %2, double %a3, i32 3
 %4 = call nnan reassoc double @llvm.vector.reduce.fmax.v4f64(<4 x double> %3)
 ret double %4
}

declare double @llvm.vector.reduce.fmax.v4f64(<4 x double>)
