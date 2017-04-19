; RUN: opt < %s -basicaa -slp-vectorizer  -S -mtriple=x86_64-unknown -mcpu=corei7-avx | FileCheck %s

%struct.complex = type { float, float }

; CHECK-LABEL: void @foo
define  void @foo (%struct.complex* %A, %struct.complex* %B, %struct.complex* %Result) {

entry:
  %0 = add i64 256, 0
  br label %loop

; CHECK-LABEL: loop
; CHECK: [[REG0:%[0-9]+]] = phi <2 x float> {{.*}}[ [[REG1:%[0-9]+]], %loop ]
; CHECK: [[REG2:%[0-9]+]] = load <2 x float>, <2 x float>*
; CHECK: [[REG3:%[0-9]+]] = fmul <2 x float> [[REG2]]
; CHECK: [[REG4:%[0-9]+]] = fmul <2 x float>
; CHECK: fsub <2 x float> [[REG3]], [[REG4]]
; CHECK: fadd <2 x float> [[REG3]], [[REG4]]
; CHECK: shufflevector <2 x float>
; CHECK: [[REG1]] = fadd <2 x float>{{.*}}[[REG0]]
loop:

  %1 = phi i64 [ 0, %entry ], [ %20, %loop ]
  %2 = phi float [ 0.000000e+00, %entry ], [ %19, %loop ]
  %3 = phi float [ 0.000000e+00, %entry ], [ %18, %loop ]
  %4 = getelementptr inbounds %"struct.complex", %"struct.complex"* %A, i64 %1, i32 0
  %5 = load float, float* %4, align 4
  %6 = getelementptr inbounds %"struct.complex", %"struct.complex"* %A, i64 %1, i32 1
  %7 = load float, float* %6, align 4
  %8 = getelementptr inbounds %"struct.complex", %"struct.complex"* %B, i64 %1, i32 0
  %9 = load float, float* %8, align 4
  %10 = getelementptr inbounds %"struct.complex", %"struct.complex"* %B, i64 %1, i32 1
  %11 = load float, float* %10, align 4
  %12 = fmul float %5, %9
  %13 = fmul float %7, %11
  %14 = fsub float %12, %13
  %15 = fmul float %7, %9
  %16 = fmul float %5, %11
  %17 = fadd float %15, %16
  %18 = fadd float %3, %14
  %19 = fadd float %2, %17
  %20 = add nuw nsw i64 %1, 1
  %21 = icmp eq i64 %20, %0
  br i1 %21, label %exit, label %loop

exit:
  %22 = getelementptr inbounds %"struct.complex", %"struct.complex"* %Result,  i32 0, i32 0
  store float %18, float* %22, align 4
  %23 = getelementptr inbounds %"struct.complex", %"struct.complex"* %Result,  i32 0, i32 1
  store float %19, float* %23, align 4

  ret void

}
