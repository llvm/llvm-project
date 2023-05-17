; RUN: opt < %s -passes=loop-vectorize -S -mcpu=pwr9 -interleave-small-loop-scalar-reduction=true 2>&1 | FileCheck %s
; RUN: opt < %s -passes='loop-vectorize' -S -mcpu=pwr9 -interleave-small-loop-scalar-reduction=true 2>&1 | FileCheck %s

; CHECK-LABEL: vector.body
; CHECK: load double, ptr
; CHECK-NEXT: load double, ptr
; CHECK-NEXT: load double, ptr
; CHECK-NEXT: load double, ptr

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

define dso_local void @test(ptr %arg, ptr %arg1) align 2 {
bb:
  %tpm15 = load ptr, ptr %arg, align 8
  %tpm19 = load ptr, ptr %arg1, align 8
  br label %bb22
bb22:                                             ; preds = %bb33, %bb
  %tpm26 = add i64 0, 1
  %tpm10 = alloca i32, align 8
  %tpm27 = getelementptr inbounds i32, ptr %tpm10, i64 %tpm26
  %tpm29 = load ptr, ptr %tpm15, align 8
  %tpm17 = alloca double, align 8
  %tpm32 = getelementptr inbounds double, ptr %tpm17, i64 %tpm26
  br label %bb40
bb33:                                             ; preds = %bb40
  %tpm37 = fsub fast double 0.000000e+00, %tpm50
  store double %tpm37, ptr %tpm19, align 8
  br label %bb22
bb40:                                             ; preds = %bb40, %bb22
  %tpm41 = phi ptr [ %tpm51, %bb40 ], [ %tpm27, %bb22 ]
  %tpm42 = phi ptr [ %tpm52, %bb40 ], [ %tpm32, %bb22 ]
  %tpm43 = phi double [ %tpm50, %bb40 ], [ 0.000000e+00, %bb22 ]
  %tpm44 = load double, ptr %tpm42, align 8
  %tpm45 = load i32, ptr %tpm41, align 4
  %tpm46 = zext i32 %tpm45 to i64
  %tpm47 = getelementptr inbounds double, ptr %tpm19, i64 %tpm46
  %tpm48 = load double, ptr %tpm47, align 8
  %tpm49 = fmul fast double %tpm48, %tpm44
  %tpm50 = fadd fast double %tpm49, %tpm43
  %tpm51 = getelementptr inbounds i32, ptr %tpm41, i64 1
  %tpm52 = getelementptr inbounds double, ptr %tpm42, i64 1
  %tpm53 = icmp eq ptr %tpm51, %tpm29
  br i1 %tpm53, label %bb33, label %bb40
}
