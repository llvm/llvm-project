; RUN: llc < %s | FileCheck %s
target triple = "powerpc64le-unknown-linux-gnu"

define void @test1(float %f, float* %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call float @llvm.minnum.f32(float %f, float 1.0)
  store float %0, float* %fp, align 4
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test1:
; CHECK: bl fminf


define void @test2(float %f, float* %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call float @llvm.maxnum.f32(float %f, float 1.0)
  store float %0, float* %fp, align 4
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test2:
; CHECK: bl fmaxf

declare float @llvm.minnum.f32(float, float)
declare float @llvm.maxnum.f32(float, float)
