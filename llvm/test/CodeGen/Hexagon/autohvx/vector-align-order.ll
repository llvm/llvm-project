; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that this doesn't crash.
; CHECK: vmem

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@g0 = external dllexport local_unnamed_addr global ptr, align 4
@g1 = external dllexport local_unnamed_addr global ptr, align 4

define dso_local fastcc void @f0(ptr %a0, i32 %a1) unnamed_addr #0 {
b0:
  %v0 = load ptr, ptr @g1, align 4
  %v1 = tail call ptr %v0(i32 16, i32 %a1, i64 2400, i32 0, i32 32)
  br i1 poison, label %b2, label %b1

b1:                                               ; preds = %b0
  %v2 = load <64 x i8>, ptr poison, align 64
  %v3 = zext <64 x i8> %v2 to <64 x i32>
  %v4 = load <128 x i8>, ptr poison, align 64
  %v5 = zext <128 x i8> %v4 to <128 x i32>
  %v6 = load <128 x i8>, ptr poison, align 64
  %v7 = zext <128 x i8> %v6 to <128 x i32>
  %v8 = getelementptr i8, ptr %a0, i32 576
  %v9 = getelementptr i8, ptr %v8, i32 -64
  %v10 = load <64 x i8>, ptr %v9, align 64
  %v11 = sext <64 x i8> %v10 to <64 x i32>
  %v12 = add nsw <64 x i32> %v11, <i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128>
  %v13 = mul nuw nsw <64 x i32> %v12, %v3
  %v14 = extractelement <64 x i32> %v13, i32 0
  store i32 %v14, ptr %v1, align 8, !tbaa !0
  %v15 = getelementptr i8, ptr %v8, i32 384
  %v16 = load <128 x i8>, ptr %v15, align 64, !tbaa !3
  %v17 = sext <128 x i8> %v16 to <128 x i32>
  %v18 = add nsw <128 x i32> %v17, <i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128>
  %v19 = mul nuw nsw <128 x i32> %v18, %v5
  %v20 = getelementptr i8, ptr %v8, i32 256
  %v21 = load <128 x i8>, ptr %v20, align 64, !tbaa !3
  %v22 = sext <128 x i8> %v21 to <128 x i32>
  %v23 = add nsw <128 x i32> %v22, <i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128>
  %v24 = mul nuw nsw <128 x i32> %v23, %v7
  %v25 = add nsw <128 x i32> %v19, %v24
  %v26 = extractelement <128 x i32> %v25, i32 0
  %v27 = insertelement <64 x i32> <i32 poison, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, i32 %v26, i64 0
  %v28 = extractelement <64 x i32> %v27, i32 0
  %v29 = getelementptr i8, ptr %v1, i32 4
  store i32 %v28, ptr %v29, align 4
  %v30 = load ptr, ptr @g0, align 4
  %v31 = call i32 %v30(ptr nonnull @f1, ptr nonnull poison, i32 0)
  unreachable

b2:                                               ; preds = %b0
  ret void
}

declare dso_local i32 @f1(i32, ptr, ptr) #0

attributes #0 = { "target-features"="+hvxv68,+hvx-length128b,+hvx-qfloat,-hvx-ieee-fp" }

!0 = !{!1, !1, i64 0}
!1 = !{!"0x555e886c7bc0", !2, i64 0}
!2 = !{!"tvm-tbaa"}
!3 = !{!4, !4, i64 0}
!4 = !{!"0x555e88cb2ce0", !2, i64 0}
