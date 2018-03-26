; XFAIL: *
; This requires further patches.
; RUN: llc -march=hexagon < %s | FileCheck %s
;
; Split all andp/orp instructions (by boosting the profitability of their
; operands, which happen to be word masks).
; This should result in a loop with two packets, but we don't generate
; post-incremented loads, so we end up with 3 packets.

; CHECK-LABEL: fred
; CHECK: loop0([[LOOP:.LBB[0-9_]+]],
; CHECK: [[LOOP]]:
; CHECK: {
; CHECK: {
; Make sure that the 3rd packet only has an add in it.
; CHECK: {
; CHECK:  r[[REG:[0-9]+]] = add(r[[REG]],#16)
; CHECK-NOT: {
; CHECK: }{{[ \t]*}}:endloop0

target triple = "hexagon"

define i32 @fred(i32 %a0, i64* nocapture readonly %a1) local_unnamed_addr #0 {
b2:
  %v3 = bitcast i64* %a1 to i32*
  %v4 = getelementptr inbounds i32, i32* %v3, i32 1
  %v5 = load i32, i32* %v3, align 4
  %v6 = load i32, i32* %v4, align 4
  %v7 = zext i32 %a0 to i64
  br label %b8

b8:                                               ; preds = %b8, %b2
  %v9 = phi i32 [ %v6, %b2 ], [ %v49, %b8 ]
  %v10 = phi i32 [ %v5, %b2 ], [ %v48, %b8 ]
  %v11 = phi i32 [ 2, %b2 ], [ %v45, %b8 ]
  %v12 = phi i64 [ 0, %b2 ], [ %v46, %b8 ]
  %v13 = phi i64 [ 0, %b2 ], [ %v47, %b8 ]
  %v14 = phi i32 [ 0, %b2 ], [ %v33, %b8 ]
  %v15 = phi i32 [ 0, %b2 ], [ %v40, %b8 ]
  %v16 = zext i32 %v10 to i64
  %v17 = or i64 %v12, %v16
  %v18 = tail call i64 @llvm.hexagon.S4.vxsubaddhr(i64 %v17, i64 %v7)
  %v19 = zext i32 %v9 to i64
  %v20 = or i64 %v13, %v19
  %v21 = tail call i64 @llvm.hexagon.S4.vxsubaddhr(i64 %v20, i64 %v7)
  %v22 = getelementptr inbounds i32, i32* %v3, i32 %v11
  %v23 = load i32, i32* %v22, align 4
  %v24 = or i32 %v11, 1
  %v25 = getelementptr inbounds i32, i32* %v3, i32 %v24
  %v26 = load i32, i32* %v25, align 4
  %v27 = zext i32 %v14 to i64
  %v28 = shl nuw i64 %v27, 32
  %v29 = zext i32 %v23 to i64
  %v30 = or i64 %v28, %v29
  %v31 = tail call i64 @llvm.hexagon.S4.vxaddsubhr(i64 %v30, i64 %v7)
  %v32 = lshr i64 %v31, 32
  %v33 = trunc i64 %v32 to i32
  %v34 = zext i32 %v15 to i64
  %v35 = shl nuw i64 %v34, 32
  %v36 = zext i32 %v26 to i64
  %v37 = or i64 %v35, %v36
  %v38 = tail call i64 @llvm.hexagon.S4.vxaddsubhr(i64 %v37, i64 %v7)
  %v39 = lshr i64 %v38, 32
  %v40 = trunc i64 %v39 to i32
  %v41 = add nuw nsw i32 %v11, 2
  %v42 = getelementptr inbounds i32, i32* %v3, i32 %v41
  %v43 = add nuw nsw i32 %v11, 3
  %v44 = getelementptr inbounds i32, i32* %v3, i32 %v43
  %v45 = add nuw nsw i32 %v11, 4
  %v46 = and i64 %v18, -4294967296
  %v47 = and i64 %v21, -4294967296
  %v48 = load i32, i32* %v42, align 4
  %v49 = load i32, i32* %v44, align 4
  %v50 = icmp ult i32 %v45, 30
  br i1 %v50, label %b8, label %b51

b51:                                              ; preds = %b8
  %v52 = zext i32 %v48 to i64
  %v53 = or i64 %v46, %v52
  %v54 = add i64 %v53, %v7
  %v55 = lshr i64 %v54, 32
  %v56 = trunc i64 %v55 to i32
  %v57 = zext i32 %v49 to i64
  %v58 = or i64 %v47, %v57
  %v59 = add i64 %v58, %v7
  %v60 = lshr i64 %v59, 32
  %v61 = trunc i64 %v60 to i32
  %v62 = tail call i64 @llvm.hexagon.A2.combinew(i32 %v33, i32 %v56)
  %v63 = lshr i64 %v62, 32
  %v64 = trunc i64 %v63 to i32
  %v65 = tail call i64 @llvm.hexagon.A2.combinew(i32 %v40, i32 %v61)
  %v66 = lshr i64 %v65, 32
  %v67 = trunc i64 %v66 to i32
  %v68 = tail call i64 @llvm.hexagon.A2.combinew(i32 %v67, i32 %v64)
  %v69 = lshr i64 %v68, 32
  %v70 = trunc i64 %v69 to i32
  ret i32 %v70
}

declare i64 @llvm.hexagon.S4.vxsubaddhr(i64, i64) #1
declare i64 @llvm.hexagon.S4.vxaddsubhr(i64, i64) #1
declare i64 @llvm.hexagon.A2.combinew(i32, i32) #1

attributes #0 = { nounwind readonly "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone }
