; REQUIRES: asserts

; RUN: opt -passes='loop-vectorize' -mtriple=x86_64-unknown-linux -S -debug %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

target triple = "x86_64-unknown-linux"

declare double @llvm.pow.f64(double, double)

; Test case where the memory runtime checks and vector body is more expensive
; than running the scalar loop.
define void @test(double* nocapture %A, double* nocapture %B, double* nocapture %C, double* nocapture %D, double* nocapture %E) {

; CHECK: Calculating cost of runtime checks:
; CHECK-NEXT:  0  for   {{.+}} = getelementptr double, double* %A, i64 16
; CHECK-NEXT:  0  for   {{.+}} = bitcast double*
; CHECK-NEXT:  0  for   {{.+}} = getelementptr double, double* %B, i64 16
; CHECK-NEXT:  0  for   {{.+}} = bitcast double*
; CHECK-NEXT:  0  for   {{.+}} = getelementptr double, double* %E, i64 16
; CHECK-NEXT:  0  for   {{.+}} = bitcast double*
; CHECK-NEXT:  0  for   {{.+}} = getelementptr double, double* %C, i64 16
; CHECK-NEXT:  0  for   {{.+}} = bitcast double*
; CHECK-NEXT:  0  for   {{.+}} = getelementptr double, double* %D, i64 16
; CHECK-NEXT:  0  for   {{.+}} = bitcast double*
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = and i1
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = and i1
; CHECK-NEXT:  1  for   {{.+}} = or i1
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = and i1
; CHECK-NEXT:  1  for   {{.+}} = or i1
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = and i1
; CHECK-NEXT:  1  for   {{.+}} = or i1
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = and i1
; CHECK-NEXT:  1  for   {{.+}} = or i1
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = and i1
; CHECK-NEXT:  1  for   {{.+}} = or i1
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = and i1
; CHECK-NEXT:  1  for   {{.+}} = or i1
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = and i1
; CHECK-NEXT:  1  for   {{.+}} = or i1
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = icmp ult i8*
; CHECK-NEXT:  1  for   {{.+}} = and i1
; CHECK-NEXT:  1  for   {{.+}} = or i1
; CHECK-NEXT: Total cost of runtime checks: 35

; CHECK: LV: Vectorization is not beneficial: expected trip count < minimum profitable VF (16 < 70)
;
; CHECK-LABEL: @test(
; CHECK-NEXT: entry:
; CHECK-NEXT:  br label %for.body
; CHECK-NOT: vector.memcheck
; CHECK-NOT: vector.body
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.A = getelementptr inbounds double, double* %A, i64 %iv
  %l.A = load double, double* %gep.A, align 4
  store double 0.0, double* %gep.A, align 4
  %p.1 = call double @llvm.pow.f64(double %l.A, double 2.0)

  %gep.B = getelementptr inbounds double, double* %B, i64 %iv
  %l.B = load double, double* %gep.B, align 4
  %p.2 = call double @llvm.pow.f64(double %l.B, double %p.1)
  store double 0.0, double* %gep.B, align 4

  %gep.C = getelementptr inbounds double, double* %C, i64 %iv
  %l.C = load double, double* %gep.C, align 4
  %p.3 = call double @llvm.pow.f64(double %p.1, double %l.C)

  %gep.D = getelementptr inbounds double, double* %D, i64 %iv
  %l.D = load double, double* %gep.D
  %p.4 = call double @llvm.pow.f64(double %p.3, double %l.D)
  %p.5 = call double @llvm.pow.f64(double %p.4, double %p.3)
  %mul = fmul double 2.0, %p.5
  %mul.2 = fmul double %mul, 2.0
  %mul.3 = fmul double %mul, %mul.2
  %gep.E = getelementptr inbounds double, double* %E, i64 %iv
  store double %mul.3, double* %gep.E, align 4
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 16
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
