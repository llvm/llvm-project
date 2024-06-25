; RUN: llc -march=hexagon -mcpu=hexagonv73 -O2 -mattr=+hvxv73,hvx-length64b  < %s | FileCheck %s
; CHECK: .p2align{{.*}}5

; Function Attrs: nounwind
define void @wobble(ptr noalias nocapture readonly %arg, i32 %arg1, i32 %arg2, i32 %arg3, ptr noalias nocapture %arg4, i32 %arg5) #0 {
bb:
  %ashr = ashr i32 %arg3, 2
  %ashr6 = ashr i32 %arg3, 1
  %add = add nsw i32 %ashr6, %ashr
  %icmp = icmp sgt i32 %arg2, 0
  br i1 %icmp, label %bb7, label %bb61

bb7:                                              ; preds = %bb
  %sdiv = sdiv i32 %arg1, 64
  %icmp8 = icmp sgt i32 %arg1, 63
  br label %bb9

bb9:                                              ; preds = %bb57, %bb7
  %phi = phi i32 [ 0, %bb7 ], [ %add58, %bb57 ]
  %ashr10 = ashr exact i32 %phi, 1
  %mul = mul nsw i32 %ashr10, %arg3
  br i1 %icmp8, label %bb11, label %bb57

bb11:                                             ; preds = %bb9
  %add12 = add nsw i32 %phi, 1
  %mul13 = mul nsw i32 %add12, %arg5
  %mul14 = mul nsw i32 %phi, %arg5
  %add15 = add i32 %add, %mul
  %add16 = add i32 %mul, %ashr
  %add17 = add i32 %mul, %ashr6
  %getelementptr = getelementptr inbounds i8, ptr %arg4, i32 %mul13
  %getelementptr18 = getelementptr inbounds i8, ptr %arg4, i32 %mul14
  %getelementptr19 = getelementptr inbounds i16, ptr %arg, i32 %add15
  %getelementptr20 = getelementptr inbounds i16, ptr %arg, i32 %add16
  %getelementptr21 = getelementptr inbounds i16, ptr %arg, i32 %add17
  %getelementptr22 = getelementptr inbounds i16, ptr %arg, i32 %mul
  %bitcast = bitcast ptr %getelementptr to ptr
  %bitcast23 = bitcast ptr %getelementptr18 to ptr
  %bitcast24 = bitcast ptr %getelementptr19 to ptr
  %bitcast25 = bitcast ptr %getelementptr20 to ptr
  %bitcast26 = bitcast ptr %getelementptr21 to ptr
  %bitcast27 = bitcast ptr %getelementptr22 to ptr
  br label %bb28

bb28:                                             ; preds = %bb28, %bb11
  %phi29 = phi i32 [ 0, %bb11 ], [ %add54, %bb28 ]
  %phi30 = phi ptr [ %bitcast27, %bb11 ], [ %getelementptr36, %bb28 ]
  %phi31 = phi ptr [ %bitcast26, %bb11 ], [ %getelementptr37, %bb28 ]
  %phi32 = phi ptr [ %bitcast25, %bb11 ], [ %getelementptr39, %bb28 ]
  %phi33 = phi ptr [ %bitcast24, %bb11 ], [ %getelementptr41, %bb28 ]
  %phi34 = phi ptr [ %bitcast, %bb11 ], [ %getelementptr53, %bb28 ]
  %phi35 = phi ptr [ %bitcast23, %bb11 ], [ %getelementptr52, %bb28 ]
  %getelementptr36 = getelementptr inbounds <16 x i32>, ptr %phi30, i32 1
  %load = load <16 x i32>, ptr %phi30, align 64, !tbaa !1
  %getelementptr37 = getelementptr inbounds <16 x i32>, ptr %phi31, i32 1
  %load38 = load <16 x i32>, ptr %phi31, align 64, !tbaa !1
  %getelementptr39 = getelementptr inbounds <16 x i32>, ptr %phi32, i32 1
  %load40 = load <16 x i32>, ptr %phi32, align 64, !tbaa !1
  %getelementptr41 = getelementptr inbounds <16 x i32>, ptr %phi33, i32 1
  %load42 = load <16 x i32>, ptr %phi33, align 64, !tbaa !1
  %call = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %load, <16 x i32> %load38)
  %call43 = tail call <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32> %load, <16 x i32> %load38)
  %call44 = tail call <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32> %load40, <16 x i32> %load42)
  %call45 = tail call <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32> %load40, <16 x i32> %load42)
  %call46 = tail call <16 x i32> @llvm.hexagon.V6.vavgh(<16 x i32> %call, <16 x i32> %call44)
  %call47 = tail call <16 x i32> @llvm.hexagon.V6.vnavgh(<16 x i32> %call, <16 x i32> %call44)
  %call48 = tail call <16 x i32> @llvm.hexagon.V6.vavgh(<16 x i32> %call43, <16 x i32> %call45)
  %call49 = tail call <16 x i32> @llvm.hexagon.V6.vnavgh(<16 x i32> %call43, <16 x i32> %call45)
  %call50 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %call47, <16 x i32> %call46)
  %call51 = tail call <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32> %call49, <16 x i32> %call48)
  %getelementptr52 = getelementptr inbounds <16 x i32>, ptr %phi35, i32 1
  store <16 x i32> %call50, ptr %phi35, align 64, !tbaa !1
  %getelementptr53 = getelementptr inbounds <16 x i32>, ptr %phi34, i32 1
  store <16 x i32> %call51, ptr %phi34, align 64, !tbaa !1
  %add54 = add nsw i32 %phi29, 1
  %icmp55 = icmp slt i32 %add54, %sdiv
  br i1 %icmp55, label %bb28, label %bb56

bb56:                                             ; preds = %bb28
  br label %bb57

bb57:                                             ; preds = %bb56, %bb9
  %add58 = add nsw i32 %phi, 2
  %icmp59 = icmp slt i32 %add58, %arg2
  br i1 %icmp59, label %bb9, label %bb60

bb60:                                             ; preds = %bb57
  br label %bb61

bb61:                                             ; preds = %bb60, %bb
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <16 x i32> @llvm.hexagon.V6.vaddh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <16 x i32> @llvm.hexagon.V6.vsubh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <16 x i32> @llvm.hexagon.V6.vavgh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <16 x i32> @llvm.hexagon.V6.vnavgh(<16 x i32>, <16 x i32>) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <16 x i32> @llvm.hexagon.V6.vsathub(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.ident = !{!0}

!0 = !{!"Clang 3.1"}
!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
