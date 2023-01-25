; RUN: llc -march=hexagon < %s | FileCheck %s

; The getelementptr's based on %a2, but with different base types caused
; a problem in vector alignment code.
; For now check that none of the vector loads are aligned.
; CHECK-NOT: = vmem(

define void @f0(ptr noalias nocapture align 64 %a0, ptr noalias nocapture readonly align 64 %a1, ptr noalias nocapture readonly align 64 %a2) #0 {
b0:
  %v0 = getelementptr float, ptr %a2, i32 74
  %v1 = getelementptr i8, ptr %a2, i32 424
  %v2 = load <32 x float>, ptr %v0, align 8
  %v3 = load <32 x float>, ptr %v1, align 8
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v4 = phi i32 [ 0, %b0 ], [ %v18, %b1 ]
  %v5 = mul nuw nsw i32 %v4, 74
  %v6 = mul nuw nsw i32 %v4, 13
  %v7 = getelementptr float, ptr %a0, i32 %v5
  %v8 = add nuw nsw i32 %v6, 1
  %v9 = getelementptr inbounds float, ptr %a1, i32 %v8
  %v10 = load float, ptr %v9, align 4
  %v11 = insertelement <32 x float> poison, float %v10, i64 0
  %v12 = shufflevector <32 x float> %v11, <32 x float> poison, <32 x i32> zeroinitializer
  %v13 = load <32 x float>, ptr %v7, align 8
  %v14 = tail call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %v12, <32 x float> %v2, <32 x float> %v13)
  store <32 x float> %v14, ptr %v7, align 8
  %v15 = getelementptr i8, ptr %v7, i32 128
  %v16 = load <32 x float>, ptr %v15, align 8
  %v17 = tail call <32 x float> @llvm.fmuladd.v32f32(<32 x float> %v12, <32 x float> %v3, <32 x float> %v16)
  store <32 x float> %v17, ptr %v15, align 8
  %v18 = add nuw nsw i32 %v4, 1
  %v19 = icmp eq i32 %v18, 11
  br i1 %v19, label %b2, label %b1

b2:                                               ; preds = %b1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <32 x float> @llvm.fmuladd.v32f32(<32 x float>, <32 x float>, <32 x float>) #1

attributes #0 = { "target-features"="+hvxv69,+hvx-length128b" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
