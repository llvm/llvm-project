; RUN: llc -march=hexagon < %s | FileCheck %s
;
; Check that this testcase doesn't crash.
; CHECK: vadd

target triple = "hexagon"

define void @fred() #0 {
b0:
  br label %b1

b1:                                               ; preds = %b7, %b0
  %v2 = phi i32 [ 0, %b0 ], [ %v16, %b7 ]
  %v3 = phi <32 x i32> [ undef, %b0 ], [ %v15, %b7 ]
  %v4 = icmp slt i32 %v2, undef
  br i1 %v4, label %b5, label %b7

b5:                                               ; preds = %b1
  %v6 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v3, <32 x i32> undef)
  br label %b7

b7:                                               ; preds = %b5, %b1
  %v8 = phi <32 x i32> [ %v6, %b5 ], [ %v3, %b1 ]
  %v9 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v8, <32 x i32> undef)
  %v10 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v9, <32 x i32> undef)
  %v11 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v10, <32 x i32> undef)
  %v12 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v11, <32 x i32> undef)
  %v13 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v12, <32 x i32> zeroinitializer)
  %v14 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v13, <32 x i32> undef)
  %v15 = tail call <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32> %v14, <32 x i32> undef)
  %v16 = add nsw i32 %v2, 8
  %v17 = icmp eq i32 %v16, 64
  br i1 %v17, label %b18, label %b1

b18:                                              ; preds = %b7
  tail call void @f0() #0
  ret void
}

declare <32 x i32> @llvm.hexagon.V6.vaddhsat.128B(<32 x i32>, <32 x i32>) #1
declare void @f0() #0

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-double" }
attributes #1 = { nounwind readnone }
