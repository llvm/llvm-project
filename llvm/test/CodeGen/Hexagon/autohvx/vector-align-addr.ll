; RUN: llc -march=hexagon -hexagon-hvx-widen=32 < %s | FileCheck %s

; Test that the Hexagon Vector Combine pass computes the address
; correctly when the loading objects that contain extra padding
; between successive objects.

; CHECK: [[REG:r[0-9]+]] = add(r{{[0-9]+}},#2432)
; CHECK: = vmem([[REG]]+#0)

define dllexport void @test(ptr %a0) local_unnamed_addr #0 {
b0:
  %v0 = add nuw nsw i32 0, 3040
  %v1 = load i8, ptr undef, align 1
  %v2 = insertelement <19 x i8> undef, i8 %v1, i32 0
  %v3 = shufflevector <19 x i8> %v2, <19 x i8> undef, <19 x i32> zeroinitializer
  %v4 = getelementptr inbounds i8, ptr %a0, i32 %v0
  %v6 = load <19 x i8>, ptr %v4, align 1
  %v7 = mul <19 x i8> %v3, %v6
  %v8 = add <19 x i8> %v7, zeroinitializer
  %v9 = add <19 x i8> zeroinitializer, %v8
  %v10 = add <19 x i8> zeroinitializer, %v9
  %v11 = add <19 x i8> zeroinitializer, %v10
  %v12 = add <19 x i8> zeroinitializer, %v11
  %v13 = add <19 x i8> zeroinitializer, %v12
  %v14 = add <19 x i8> zeroinitializer, %v13
  %v15 = add <19 x i8> zeroinitializer, %v14
  %v16 = add <19 x i8> zeroinitializer, %v15
  %v17 = add <19 x i8> zeroinitializer, %v16
  %v18 = add <19 x i8> zeroinitializer, %v17
  %v19 = load i8, ptr undef, align 1
  %v20 = insertelement <19 x i8> undef, i8 %v19, i32 0
  %v21 = shufflevector <19 x i8> %v20, <19 x i8> undef, <19 x i32> zeroinitializer
  %v22 = add nuw nsw i32 0, 5472
  %v23 = getelementptr inbounds i8, ptr %a0, i32 %v22
  %v25 = load <19 x i8>, ptr %v23, align 1
  %v26 = mul <19 x i8> %v21, %v25
  %v27 = add <19 x i8> %v26, %v18
  %v28 = add <19 x i8> zeroinitializer, %v27
  %v29 = add <19 x i8> zeroinitializer, %v28
  %v30 = add <19 x i8> zeroinitializer, %v29
  store <19 x i8> %v30, ptr %a0, align 1
  ret void
}

attributes #0 = { "target-features"="+hvxv66,+hvx-length128b" }
