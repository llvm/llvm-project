; RUN: llc -march=hexagon < %s | FileCheck %s

; Check for a sane output. This testcase used to cause a crash.
; CHECK: vlut16

target triple = "hexagon-unknown--elf"

declare void @halide_malloc() local_unnamed_addr #0

declare <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32>, <32 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vmpyiewuh.128B(<32 x i32>, <32 x i32>) #1
declare <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32>, <64 x i32>) #1
declare <32 x i32> @llvm.hexagon.V6.vasrwhsat.128B(<32 x i32>, <32 x i32>, i32) #1
declare <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32>, <32 x i32>, i32) #1
declare <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32>, <32 x i32>, <32 x i32>, i32) #1

define hidden void @fred(<32 x i32>* %a0, i32 %a1) #0 {
b0:
  %v1 = ashr i32 %a1, 7
  %v2 = shl nsw i32 %v1, 7
  switch i32 undef, label %b7 [
    i32 1, label %b3
    i32 2, label %b5
    i32 3, label %b6
  ]

b3:                                               ; preds = %b0
  unreachable

b4:                                               ; preds = %b7
  switch i32 undef, label %b9 [
    i32 1, label %b8
    i32 2, label %b10
    i32 3, label %b11
  ]

b5:                                               ; preds = %b0
  unreachable

b6:                                               ; preds = %b0
  unreachable

b7:                                               ; preds = %b0
  br label %b4

b8:                                               ; preds = %b4
  br label %b12

b9:                                               ; preds = %b4
  br label %b12

b10:                                              ; preds = %b4
  br label %b12

b11:                                              ; preds = %b4
  br label %b12

b12:                                              ; preds = %b11, %b10, %b9, %b8
  br label %b13

b13:                                              ; preds = %b14, %b12
  br label %b14

b14:                                              ; preds = %b13
  br i1 undef, label %b15, label %b13

b15:                                              ; preds = %b14
  br label %b16

b16:                                              ; preds = %b15
  br i1 undef, label %b17, label %b18

b17:                                              ; preds = %b16
  unreachable

b18:                                              ; preds = %b16
  tail call void @halide_malloc()
  br label %b19

b19:                                              ; preds = %b18
  %v21 = icmp sgt i32 %a1, 0
  br i1 %v21, label %b20, label %b21

b20:                                              ; preds = %b19
  br label %b32

b21:                                              ; preds = %b38, %b19
  %v22 = zext i32 %v2 to i64
  %v23 = lshr i64 %v22, 31
  %v24 = shl nuw nsw i64 %v23, 1
  %v25 = or i64 %v24, 0
  %v26 = icmp ult i64 %v23, 2147483648
  %v27 = mul nuw nsw i64 %v25, 3
  %v28 = add nuw nsw i64 %v27, 0
  %v29 = and i64 %v28, 133143986176
  %v30 = icmp eq i64 %v29, 0
  %v31 = and i1 %v26, %v30
  br label %b39

b32:                                              ; preds = %b20
  %v33 = zext i32 %v2 to i64
  %v34 = mul nuw nsw i64 %v33, 12
  %v35 = icmp ult i64 %v34, 2147483648
  %v36 = and i1 %v35, undef
  br i1 %v36, label %b38, label %b37

b37:                                              ; preds = %b32
  ret void

b38:                                              ; preds = %b32
  tail call void @halide_malloc()
  br label %b21

b39:                                              ; preds = %b42, %b21
  br label %b40

b40:                                              ; preds = %b39
  br i1 %v31, label %b42, label %b41

b41:                                              ; preds = %b40
  unreachable

b42:                                              ; preds = %b40
  %v43 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.128B(<32 x i32> undef, <32 x i32> undef, i32 0)
  %v44 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %v43, <32 x i32> undef, <32 x i32> undef, i32 1)
  %v45 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %v44, <32 x i32> undef, <32 x i32> undef, i32 2)
  %v46 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %v45, <32 x i32> undef, <32 x i32> undef, i32 3)
  %v47 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %v46, <32 x i32> undef, <32 x i32> undef, i32 4)
  %v48 = tail call <64 x i32> @llvm.hexagon.V6.vlutvwh.oracc.128B(<64 x i32> %v47, <32 x i32> undef, <32 x i32> undef, i32 5)
  %v49 = tail call <32 x i32> @llvm.hexagon.V6.lo.128B(<64 x i32> %v48)
  %v50 = tail call <32 x i32> @llvm.hexagon.V6.vmpyiewuh.128B(<32 x i32> undef, <32 x i32> %v49) #2
  %v51 = tail call <64 x i32> @llvm.hexagon.V6.vcombine.128B(<32 x i32> undef, <32 x i32> %v50) #2
  %v52 = tail call <64 x i32> @llvm.hexagon.V6.vaddw.dv.128B(<64 x i32> %v51, <64 x i32> undef) #2
  %v53 = tail call <32 x i32> @llvm.hexagon.V6.hi.128B(<64 x i32> %v52) #2
  %v54 = tail call <32 x i32> @llvm.hexagon.V6.vasrwhsat.128B(<32 x i32> %v53, <32 x i32> undef, i32 15) #2
  store <32 x i32> %v54, <32 x i32>* %a0, align 128
  br label %b39
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
