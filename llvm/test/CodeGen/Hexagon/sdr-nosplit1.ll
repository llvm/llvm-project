; RUN: llc -mtriple=hexagon < %s | FileCheck %s
;
; Make sure that the A2_andp is not split.
;
; CHECK: loop0([[LOOP:.LBB[_0-9]+]],{{.*}})
; CHECK: [[LOOP]]:
; CHECK: and(r{{[0-9]+}}:{{[0-9]+}},r{{[0-9]+}}:{{[0-9]+}})

target triple = "hexagon"

define void @fred(i64 %a0, i64 %a1, i64 %a2, ptr nocapture %a3, i32 %a4) local_unnamed_addr #0 {
b5:
  %v6 = icmp sgt i32 %a4, 0
  br i1 %v6, label %b7, label %b20

b7:                                               ; preds = %b7, %b5
  %v8 = phi ptr [ %v16, %b7 ], [ %a3, %b5 ]
  %v9 = phi i32 [ %v18, %b7 ], [ 0, %b5 ]
  %v10 = phi i64 [ %v17, %b7 ], [ %a0, %b5 ]
  %v11 = tail call i64 @llvm.hexagon.A2.andp(i64 %v10, i64 1085102592571150095)
  %v12 = tail call i32 @llvm.hexagon.A2.vcmpbgtu(i64 %a1, i64 %v11)
  %v13 = tail call i64 @llvm.hexagon.A2.vsubub(i64 %v11, i64 %a1)
  %v14 = and i32 %v12, 255
  %v15 = tail call i64 @llvm.hexagon.C2.vmux(i32 %v14, i64 %a2, i64 %v13)
  store i64 %v15, ptr %v8, align 8
  %v16 = getelementptr i64, ptr %v8, i32 1
  %v17 = load i64, ptr %v16, align 8
  %v18 = add nuw nsw i32 %v9, 1
  %v19 = icmp eq i32 %v18, %a4
  br i1 %v19, label %b20, label %b7

b20:                                              ; preds = %b7, %b5
  ret void
}

declare i64 @llvm.hexagon.A2.andp(i64, i64) #1
declare i32 @llvm.hexagon.A2.vcmpbgtu(i64, i64) #1
declare i64 @llvm.hexagon.A2.vsubub(i64, i64) #1
declare i64 @llvm.hexagon.C2.vmux(i32, i64, i64) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone }
