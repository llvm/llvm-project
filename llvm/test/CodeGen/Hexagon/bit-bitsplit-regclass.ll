; RUN: llc -march=hexagon -hexagon-initial-cfg-cleanup=0 -hexagon-instsimplify=0 < %s | FileCheck %s

; Check for successful compilation.
; CHECK: r{{[0-9]+}} = insert(r{{[0-9]+}},#1,#31)

; This cannot be a .mir test, because the failure depends on ordering of
; virtual registers, and the .mir loader renumbers them in a way that hides
; the problem.

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; Function Attrs: nounwind
define void @f0() #0 align 2 {
b0:
  br label %b1

b1:                                               ; preds = %b3, %b0
  %v0 = phi i64 [ 0, %b0 ], [ %v6, %b3 ]
  br i1 undef, label %b2, label %b3

b2:                                               ; preds = %b1
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v1 = phi i64 [ undef, %b2 ], [ %v0, %b1 ]
  %v2 = and i64 %v1, 1
  %v3 = trunc i64 %v2 to i32
  %v4 = tail call i32 @llvm.hexagon.C2.mux(i32 %v3, i32 undef, i32 undef)
  %v5 = trunc i32 %v4 to i8
  store i8 %v5, ptr undef, align 1
  %v6 = lshr i64 %v1, 1
  br label %b1
}

; Function Attrs: nounwind readnone
declare i32 @llvm.hexagon.C2.mux(i32, i32, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone }
