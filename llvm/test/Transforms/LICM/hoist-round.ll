; RUN: opt -S -licm < %s | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' -S %s | FileCheck %s

target datalayout = "E-m:e-p:32:32-i8:8:8-i16:16:16-i64:32:32-f64:32:32-v64:32:32-v128:32:32-a0:0:32-n32"

; This test verifies that ceil, floor, nearbyint, trunc, rint, round,
; copysign, minnum, maxnum and fabs intrinsics are considered safe
; to speculate.

; CHECK-LABEL: @test
; CHECK: call float @llvm.ceil.f32
; CHECK: call float @llvm.floor.f32
; CHECK: call float @llvm.nearbyint.f32
; CHECK: call float @llvm.rint.f32
; CHECK: call float @llvm.round.f32
; CHECK: call float @llvm.trunc.f32
; CHECK: call float @llvm.fabs.f32
; CHECK: call float @llvm.copysign.f32
; CHECK: call float @llvm.minnum.f32
; CHECK: call float @llvm.maxnum.f32
; CHECK: call float @llvm.powi.f32
; CHECK: for.body:

define void @test(float %arg1, float %arg2) {
entry:
  br label %for.head

for.head:
  %IND = phi i32 [ 0, %entry ], [ %IND.new, %for.body ]
  %CMP = icmp slt i32 %IND, 10
  br i1 %CMP, label %for.body, label %exit

for.body:
  %tmp.1 = call float @llvm.ceil.f32(float %arg1)
  %tmp.2 = call float @llvm.floor.f32(float %tmp.1)
  %tmp.3 = call float @llvm.nearbyint.f32(float %tmp.2)
  %tmp.4 = call float @llvm.rint.f32(float %tmp.3)
  %tmp.5 = call float @llvm.round.f32(float %tmp.4)
  %tmp.6 = call float @llvm.trunc.f32(float %tmp.5)
  %tmp.7 = call float @llvm.fabs.f32(float %tmp.6)
  %tmp.8 = call float @llvm.copysign.f32(float %tmp.7, float %arg2)
  %tmp.9 = call float @llvm.minnum.f32(float %tmp.8, float %arg2)
  %tmp.10 = call float @llvm.maxnum.f32(float %tmp.9, float %arg2)
  %tmp.11 = call float @llvm.powi.f32(float %tmp.10, i32 4)
  call void @consume(float %tmp.11)
  %IND.new = add i32 %IND, 1
  br label %for.head

exit:
  ret void
}

declare void @consume(float)

declare float @llvm.ceil.f32(float)
declare float @llvm.floor.f32(float)
declare float @llvm.nearbyint.f32(float)
declare float @llvm.rint.f32(float)
declare float @llvm.round.f32(float)
declare float @llvm.trunc.f32(float)
declare float @llvm.fabs.f32(float)
declare float @llvm.copysign.f32(float, float)
declare float @llvm.minnum.f32(float, float)
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.powi.f32(float, i32)
