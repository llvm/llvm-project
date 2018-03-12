; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that 3 or more addressing modes using the same constant extender are
; transformed into using a register.
; CHECK: r{{[0-9]+}} = ##g1
; CHECK: r{{[0-9]+}} = memw(r{{[0-9]+}}+r{{[0-9]+}}<<#2)
; CHECK: r{{[0-9]+}} = memw(r{{[0-9]+}}+r{{[0-9]+}}<<#2)
; CHECK: r{{[0-9]+}} = memw(r{{[0-9]+}}+r{{[0-9]+}}<<#2)
; CHECK-NOT: r{{[0-9]+}} = memw(r{{[0-9]+}}<<#2+##g1)
; CHECK-NOT: r{{[0-9]+}} = memw(r{{[0-9]+}}<<#2+##g1)
; CHECK-NOT: r{{[0-9]+}} = memw(r{{[0-9]+}}<<#2+##g1)
; CHECK:  memw(r{{[0-9]+}}+r{{[0-9]+}}<<#2) = r{{[0-9]+}}
; CHECK:  memw(r{{[0-9]+}}+r{{[0-9]+}}<<#2) = r{{[0-9]+}}
; CHECK:  memw(r{{[0-9]+}}+r{{[0-9]+}}<<#2) = r{{[0-9]+}}
; CHECK-NOT:  memw(r{{[0-9]+}}<<#2+##g1) = r{{[0-9]+}}
; CHECK-NOT:  memw(r{{[0-9]+}}<<#2+##g1) = r{{[0-9]+}}
; CHECK-NOT:  memw(r{{[0-9]+}}<<#2+##g1) = r{{[0-9]+}}

target triple = "hexagon-unknown-linux-gnu"

@g0 = external global i32
@g1 = external global [13595 x i32], align 8
@g2 = external global [13595 x i32], align 8

define i32 @f0(i32 %a0, i32* nocapture %a1) {
b0:
  %v0 = load i32, i32* %a1, align 4
  %v1 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g1, i32 0, i32 %v0
  %v2 = load i32, i32* %v1, align 4
  %v3 = icmp sgt i32 %v2, %a0
  br i1 %v3, label %b1, label %b2

b1:                                               ; preds = %b0
  %v4 = load i32, i32* @g0, align 4
  store i32 %v4, i32* %a1, align 4
  %v5 = load i32, i32* @g0, align 4
  %v6 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g2, i32 0, i32 %v5
  %v7 = load i32, i32* %v6, align 4
  store i32 %v7, i32* @g0, align 4
  %v8 = load i32, i32* %a1, align 4
  %v9 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g2, i32 0, i32 %v8
  store i32 %v0, i32* %v9, align 4
  %v10 = load i32, i32* %a1, align 4
  %v11 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g1, i32 0, i32 %v10
  store i32 %a0, i32* %v11, align 4
  br label %b16

b2:                                               ; preds = %b0
  %v12 = icmp eq i32 %v2, %a0
  br i1 %v12, label %b16, label %b3

b3:                                               ; preds = %b2
  br label %b4

b4:                                               ; preds = %b13, %b3
  %v13 = phi i32 [ %v45, %b13 ], [ %v0, %b3 ]
  %v14 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g2, i32 0, i32 %v13
  %v15 = load i32, i32* %v14, align 4
  %v16 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g1, i32 0, i32 %v15
  %v17 = load i32, i32* %v16, align 4
  %v18 = icmp slt i32 %v17, %a0
  br i1 %v18, label %b7, label %b5

b5:                                               ; preds = %b4
  %v19 = icmp eq i32 %v17, %a0
  br i1 %v19, label %b16, label %b6

b6:                                               ; preds = %b5
  %v20 = load i32, i32* @g0, align 4
  store i32 %v20, i32* %v14, align 4
  %v21 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g1, i32 0, i32 %v20
  store i32 %a0, i32* %v21, align 4
  %v22 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g2, i32 0, i32 %v20
  %v23 = load i32, i32* %v22, align 4
  store i32 %v23, i32* @g0, align 4
  store i32 %v15, i32* %v22, align 4
  br label %b16

b7:                                               ; preds = %b4
  %v24 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g2, i32 0, i32 %v15
  %v25 = load i32, i32* %v24, align 4
  %v26 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g1, i32 0, i32 %v25
  %v27 = load i32, i32* %v26, align 4
  %v28 = icmp slt i32 %v27, %a0
  br i1 %v28, label %b10, label %b8

b8:                                               ; preds = %b7
  %v29 = icmp eq i32 %v27, %a0
  br i1 %v29, label %b16, label %b9

b9:                                               ; preds = %b8
  %v30 = load i32, i32* @g0, align 4
  store i32 %v30, i32* %v24, align 4
  %v31 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g1, i32 0, i32 %v30
  store i32 %a0, i32* %v31, align 4
  %v32 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g2, i32 0, i32 %v30
  %v33 = load i32, i32* %v32, align 4
  store i32 %v33, i32* @g0, align 4
  store i32 %v25, i32* %v32, align 4
  br label %b16

b10:                                              ; preds = %b7
  %v34 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g2, i32 0, i32 %v25
  %v35 = load i32, i32* %v34, align 4
  %v36 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g1, i32 0, i32 %v35
  %v37 = load i32, i32* %v36, align 4
  %v38 = icmp slt i32 %v37, %a0
  br i1 %v38, label %b13, label %b11

b11:                                              ; preds = %b10
  %v39 = icmp eq i32 %v37, %a0
  br i1 %v39, label %b16, label %b12

b12:                                              ; preds = %b11
  %v40 = load i32, i32* @g0, align 4
  store i32 %v40, i32* %v34, align 4
  %v41 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g1, i32 0, i32 %v40
  store i32 %a0, i32* %v41, align 4
  %v42 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g2, i32 0, i32 %v40
  %v43 = load i32, i32* %v42, align 4
  store i32 %v43, i32* @g0, align 4
  store i32 %v35, i32* %v42, align 4
  br label %b16

b13:                                              ; preds = %b10
  %v44 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g2, i32 0, i32 %v35
  %v45 = load i32, i32* %v44, align 4
  %v46 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g1, i32 0, i32 %v45
  %v47 = load i32, i32* %v46, align 4
  %v48 = icmp slt i32 %v47, %a0
  br i1 %v48, label %b4, label %b14

b14:                                              ; preds = %b13
  %v49 = icmp eq i32 %v47, %a0
  br i1 %v49, label %b16, label %b15

b15:                                              ; preds = %b14
  %v50 = load i32, i32* @g0, align 4
  store i32 %v50, i32* %v44, align 4
  %v51 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g1, i32 0, i32 %v50
  store i32 %a0, i32* %v51, align 4
  %v52 = getelementptr inbounds [13595 x i32], [13595 x i32]* @g2, i32 0, i32 %v50
  %v53 = load i32, i32* %v52, align 4
  store i32 %v53, i32* @g0, align 4
  store i32 %v45, i32* %v52, align 4
  br label %b16

b16:                                              ; preds = %b15, %b14, %b12, %b11, %b9, %b8, %b6, %b5, %b2, %b1
  %v54 = phi i32 [ 1, %b1 ], [ 1, %b6 ], [ 1, %b9 ], [ 1, %b12 ], [ 1, %b15 ], [ 0, %b2 ], [ 0, %b5 ], [ 0, %b8 ], [ 0, %b11 ], [ 0, %b14 ]
  ret i32 %v54
}
