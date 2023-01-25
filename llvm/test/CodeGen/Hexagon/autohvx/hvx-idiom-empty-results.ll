; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that this doesn't crash.
; CHECK: func_end

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define void @f0(ptr %a0, ptr %a1, ptr %a2) unnamed_addr #0 {
b0:
  %v0 = load i8, ptr %a0, align 1
  %v1 = load i32, ptr %a1, align 4
  %v2 = load i32, ptr %a2, align 4
  %v3 = zext i8 %v0 to i32
  %v4 = getelementptr inbounds i8, ptr null, i32 %v1
  %v5 = add nsw i32 %v2, 2
  %v8 = insertelement <16 x i32> poison, i32 %v3, i64 0
  %v9 = shufflevector <16 x i32> %v8, <16 x i32> poison, <16 x i32> zeroinitializer
  br label %b1

b1:                                               ; preds = %b3, %b2
  %v10 = phi ptr [ %v4, %b0 ], [ %v19, %b1 ]
  %v11 = add nsw <16 x i32> zeroinitializer, <i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128, i32 -128>
  %v12 = mul nsw <16 x i32> %v11, %v9
  %v13 = add nsw <16 x i32> %v12, <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  %v14 = ashr <16 x i32> %v13, <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %v15 = tail call <16 x i32> @llvm.smin.v16i32(<16 x i32> %v14, <16 x i32> <i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127, i32 127>)
  %v16 = add nsw <16 x i32> %v15, <i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128, i32 128>
  %v17 = select <16 x i1> zeroinitializer, <16 x i32> zeroinitializer, <16 x i32> %v16
  %v18 = trunc <16 x i32> %v17 to <16 x i8>
  %v19 = getelementptr inbounds i8, ptr %v10, i32 1
  store <16 x i8> %v18, ptr %v19, align 1
  br label %b1, !llvm.loop !0
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <16 x i32> @llvm.smin.v16i32(<16 x i32>, <16 x i32>) #1

attributes #0 = { "target-features"="+hvx-length64b,+hvxv66,+v66,-long-calls,-small-data" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}
