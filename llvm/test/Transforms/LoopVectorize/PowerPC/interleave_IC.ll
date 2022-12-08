; RUN: opt < %s -passes=loop-vectorize -S -mcpu=pwr9 -interleave-small-loop-scalar-reduction=true 2>&1 | FileCheck %s
; RUN: opt < %s -passes='loop-vectorize' -S -mcpu=pwr9 -interleave-small-loop-scalar-reduction=true 2>&1 | FileCheck %s

; CHECK-LABEL: vector.body
; CHECK: load double, double*
; CHECK-NEXT: load double, double*
; CHECK-NEXT: load double, double*
; CHECK-NEXT: load double, double*

; CHECK: fmul fast double
; CHECK-NEXT: fmul fast double
; CHECK-NEXT: fmul fast double
; CHECK-NEXT: fmul fast double

; CHECK: fadd fast double
; CHECK-NEXT: fadd fast double
; CHECK-NEXT: fadd fast double
; CHECK-NEXT: fadd fast double

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

define dso_local void @test(i32*** %arg, double** %arg1) align 2 {
bb:
  %tpm15 = load i32**, i32*** %arg, align 8
  %tpm19 = load double*, double** %arg1, align 8
  br label %bb22
bb22:                                             ; preds = %bb33, %bb
  %tpm26 = add i64 0, 1
  %tpm10 = alloca i32, align 8
  %tpm27 = getelementptr inbounds i32, i32* %tpm10, i64 %tpm26
  %tpm28 = getelementptr inbounds i32*, i32** %tpm15, i64 0
  %tpm29 = load i32*, i32** %tpm28, align 8
  %tpm17 = alloca double, align 8
  %tpm32 = getelementptr inbounds double, double* %tpm17, i64 %tpm26
  br label %bb40
bb33:                                             ; preds = %bb40
  %tpm35 = getelementptr inbounds double, double* %tpm19, i64 0
  %tpm37 = fsub fast double 0.000000e+00, %tpm50
  store double %tpm37, double* %tpm35, align 8
  br label %bb22
bb40:                                             ; preds = %bb40, %bb22
  %tpm41 = phi i32* [ %tpm51, %bb40 ], [ %tpm27, %bb22 ]
  %tpm42 = phi double* [ %tpm52, %bb40 ], [ %tpm32, %bb22 ]
  %tpm43 = phi double [ %tpm50, %bb40 ], [ 0.000000e+00, %bb22 ]
  %tpm44 = load double, double* %tpm42, align 8
  %tpm45 = load i32, i32* %tpm41, align 4
  %tpm46 = zext i32 %tpm45 to i64
  %tpm47 = getelementptr inbounds double, double* %tpm19, i64 %tpm46
  %tpm48 = load double, double* %tpm47, align 8
  %tpm49 = fmul fast double %tpm48, %tpm44
  %tpm50 = fadd fast double %tpm49, %tpm43
  %tpm51 = getelementptr inbounds i32, i32* %tpm41, i64 1
  %tpm52 = getelementptr inbounds double, double* %tpm42, i64 1
  %tpm53 = icmp eq i32* %tpm51, %tpm29
  br i1 %tpm53, label %bb33, label %bb40
}
