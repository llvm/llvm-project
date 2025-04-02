; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; During lowering, a BUILD_VECTOR of undef values was created. This was
; not properly handled by buildHvxVectorReg, which tried to generate a
; splat, but had no source value.
;
; Check that this compiles successfully.
; CHECK: vsplat

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@g0 = global <32 x i8> zeroinitializer

define void @fred(ptr %a0) #0 {
b0:
  %v1 = load i8, ptr %a0, align 1
  %v2 = insertelement <32 x i8> undef, i8 %v1, i32 31
  %v3 = zext <32 x i8> %v2 to <32 x i16>
  %v4 = add nuw nsw <32 x i16> %v3, zeroinitializer
  %v5 = add nuw nsw <32 x i16> %v4, zeroinitializer
  %v6 = add nuw nsw <32 x i16> %v5, zeroinitializer
  %v7 = add nuw nsw <32 x i16> %v6, zeroinitializer
  %v8 = add nuw nsw <32 x i16> %v7, zeroinitializer
  %v9 = add nuw nsw <32 x i16> %v8, zeroinitializer
  %v10 = add <32 x i16> %v9, zeroinitializer
  %v11 = add <32 x i16> %v10, zeroinitializer
  %v12 = add <32 x i16> %v11, zeroinitializer
  %v13 = add <32 x i16> %v12, zeroinitializer
  %v14 = add <32 x i16> %v13, zeroinitializer
  %v15 = add <32 x i16> %v14, zeroinitializer
  %v16 = add <32 x i16> %v15, zeroinitializer
  %v17 = add <32 x i16> %v16, zeroinitializer
  %v18 = add <32 x i16> %v17, zeroinitializer
  %v19 = lshr <32 x i16> %v18, <i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4>
  %v20 = trunc <32 x i16> %v19 to <32 x i8>
  store <32 x i8> %v20, ptr @g0, align 1
  ret void
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
