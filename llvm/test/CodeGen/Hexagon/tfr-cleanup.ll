; RUN: llc -march=hexagon -O3 -hexagon-eif=0 < %s | FileCheck %s
; Without TFR cleanup, the last block contained
; {
;   r3 = xor(r1, r2)
;   r1 = #0
; }
; {
;   r7 = r1
;   r0 = zxtb(r3)
; }
; After TFR cleanup, the copy "r7 = r1" should be simplified to "r7 = #0".
; There shouldn't be any register copies in that block anymore.
;
; CHECK: LBB0_5:
; CHECK-NOT: r{{[0-9]+}} = r{{[0-9]+}}

target triple = "hexagon"

; Function Attrs: nounwind readnone
define i64 @f0(i64 %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = trunc i64 %a0 to i32
  %v1 = lshr i64 %a0, 32
  %v2 = trunc i64 %v1 to i32
  %v3 = lshr i64 %a0, 40
  %v4 = lshr i64 %a0, 48
  %v5 = trunc i64 %v4 to i16
  %v6 = icmp sgt i32 %a2, %a1
  %v7 = lshr i32 %v0, 10
  br i1 %v6, label %b1, label %b2

b1:                                               ; preds = %b0
  %v8 = add nsw i32 %v7, 4190971
  %v9 = and i32 %v8, 4194303
  %v10 = shl nuw nsw i64 %v3, 24
  %v11 = trunc i64 %v10 to i32
  %v12 = ashr exact i32 %v11, 24
  %v13 = or i32 %v12, 102
  br label %b3

b2:                                               ; preds = %b0
  %v14 = add nsw i32 %v7, 4189760
  %v15 = trunc i64 %v3 to i32
  %v16 = or i32 %v15, 119
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v17 = phi i32 [ %v13, %b1 ], [ %v16, %b2 ]
  %v18 = phi i32 [ %v9, %b1 ], [ %v14, %b2 ]
  %v19 = shl i32 %v18, 10
  %v20 = icmp sgt i32 %a1, %a2
  br i1 %v20, label %b4, label %b5

b4:                                               ; preds = %b3
  %v21 = and i32 %v2, 140
  %v22 = or i32 %v21, 115
  %v23 = and i16 %v5, 12345
  br label %b6

b5:                                               ; preds = %b3
  %v24 = xor i32 %v2, 23
  %v25 = or i16 %v5, 12345
  br label %b6

b6:                                               ; preds = %b5, %b4
  %v26 = phi i16 [ %v23, %b4 ], [ %v25, %b5 ]
  %v27 = phi i32 [ %v22, %b4 ], [ %v24, %b5 ]
  %v28 = zext i16 %v26 to i64
  %v29 = shl nuw i64 %v28, 48
  %v30 = and i32 %v27, 255
  %v31 = zext i32 %v30 to i64
  %v32 = shl nuw nsw i64 %v31, 40
  %v33 = and i32 %v17, 255
  %v34 = zext i32 %v33 to i64
  %v35 = shl nuw nsw i64 %v34, 32
  %v36 = zext i32 %v19 to i64
  %v37 = or i64 %v36, %v35
  %v38 = or i64 %v37, %v29
  %v39 = or i64 %v38, %v32
  ret i64 %v39
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv55" }
