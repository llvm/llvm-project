; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: f1

target triple = "hexagon"

%s.0 = type { i32 }

@g0 = internal unnamed_addr global ptr null, section ".data.............", align 4
@g1 = internal global i32 0, section ".data.............", align 4

; Function Attrs: nounwind
define ptr @f0(ptr %a0) #0 {
b0:
  %v0 = getelementptr inbounds i32, ptr %a0, i32 -1
  %v1 = load i32, ptr %v0, align 4
  %v2 = and i32 %v1, -3
  store i32 %v2, ptr %v0, align 4
  %v3 = getelementptr inbounds i32, ptr %a0, i32 -2
  %v4 = load i32, ptr %v3, align 4
  %v5 = lshr i32 %v4, 2
  %v6 = xor i32 %v5, -1
  %v7 = getelementptr inbounds i32, ptr %a0, i32 %v6
  %v8 = lshr i32 %v1, 2
  %v9 = add i32 %v8, -1
  %v10 = getelementptr inbounds i32, ptr %a0, i32 %v9
  %v11 = load i32, ptr %v10, align 4
  %v12 = lshr i32 %v11, 2
  %v13 = icmp eq i32 %v12, 0
  br i1 %v13, label %b3, label %b1

b1:                                               ; preds = %b0
  %v14 = add i32 %v12, %v9
  %v15 = getelementptr inbounds i32, ptr %a0, i32 %v14
  %v16 = load i32, ptr %v15, align 4
  %v17 = and i32 %v16, 1
  %v18 = icmp eq i32 %v17, 0
  br i1 %v18, label %b3, label %b2

b2:                                               ; preds = %b1
  %v19 = add nsw i32 %v12, %v8
  %v20 = shl i32 %v19, 2
  %v21 = and i32 %v1, 1
  %v22 = or i32 %v20, %v21
  store i32 %v22, ptr %v0, align 4
  br label %b3

b3:                                               ; preds = %b2, %b1, %b0
  %v23 = phi i32 [ %v2, %b1 ], [ %v2, %b0 ], [ %v22, %b2 ]
  %v24 = and i32 %v23, 1
  %v25 = icmp eq i32 %v24, 0
  br i1 %v25, label %b5, label %b4

b4:                                               ; preds = %b3
  %v26 = load i32, ptr %v7, align 4
  %v27 = and i32 %v26, -4
  %v28 = add i32 %v27, %v23
  %v29 = and i32 %v28, -4
  %v30 = and i32 %v26, 3
  %v31 = or i32 %v29, %v30
  store i32 %v31, ptr %v7, align 4
  br label %b5

b5:                                               ; preds = %b4, %b3
  %v32 = phi i32 [ %v31, %b4 ], [ %v23, %b3 ]
  %v33 = phi ptr [ %v7, %b4 ], [ %v0, %b3 ]
  %v35 = lshr i32 %v32, 2
  %v36 = add i32 %v35, -1
  %v37 = getelementptr inbounds %s.0, ptr %v33, i32 %v36, i32 0
  %v38 = load i32, ptr %v37, align 4
  %v39 = shl nuw i32 %v35, 2
  %v40 = and i32 %v38, 3
  %v41 = or i32 %v40, %v39
  store i32 %v41, ptr %v37, align 4
  %v42 = load i32, ptr %v33, align 4
  %v43 = lshr i32 %v42, 2
  %v44 = getelementptr inbounds %s.0, ptr %v33, i32 %v43, i32 0
  %v45 = load i32, ptr %v44, align 4
  %v46 = or i32 %v45, 1
  store i32 %v46, ptr %v44, align 4
  ret ptr %v33
}

; Function Attrs: nounwind
define i64 @f1(i32 %a0) #0 {
b0:
  %v0 = load ptr, ptr @g0, align 4, !tbaa !0
  %v1 = getelementptr inbounds %s.0, ptr %v0, i32 7
  tail call void @f2(ptr @g1) #0
  br label %b1

b1:                                               ; preds = %b5, %b0
  %v2 = phi ptr [ %v1, %b0 ], [ %v20, %b5 ]
  %v4 = load i32, ptr %v2, align 4
  %v5 = and i32 %v4, 2
  %v6 = icmp eq i32 %v5, 0
  br i1 %v6, label %b3, label %b2

b2:                                               ; preds = %b1
  tail call fastcc void @f8()
  %v7 = getelementptr inbounds %s.0, ptr %v2, i32 1, i32 0
  %v8 = tail call ptr @f0(ptr %v7)
  tail call fastcc void @f7()
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v9 = phi ptr [ %v8, %b2 ], [ %v2, %b1 ]
  %v11 = load i32, ptr %v9, align 4
  %v12 = lshr i32 %v11, 2
  %v13 = getelementptr inbounds %s.0, ptr %v9, i32 %v12, i32 0
  %v14 = load i32, ptr %v13, align 4
  %v15 = and i32 %v14, 1
  %v16 = icmp eq i32 %v15, 0
  br i1 %v16, label %b5, label %b4

b4:                                               ; preds = %b3
  %v17 = mul i32 %v12, 4
  %v18 = add i32 %v17, -4
  %v19 = icmp ult i32 %v18, %a0
  br i1 %v19, label %b5, label %b7

b5:                                               ; preds = %b4, %b3
  %v20 = getelementptr inbounds %s.0, ptr %v9, i32 %v12
  %v21 = icmp ult i32 %v14, 4
  br i1 %v21, label %b6, label %b1

b6:                                               ; preds = %b5
  tail call fastcc void @f3()
  br label %b11

b7:                                               ; preds = %b4
  %v22 = add i32 %a0, 4
  %v23 = lshr i32 %v22, 2
  %v24 = add i32 %v23, 8
  %v25 = lshr i32 %v24, 3
  %v26 = mul nsw i32 %v25, 8
  %v27 = sub nsw i32 %v12, %v26
  %v28 = icmp sgt i32 %v27, 7
  br i1 %v28, label %b8, label %b9

b8:                                               ; preds = %b7
  %v29 = getelementptr inbounds %s.0, ptr %v9, i32 %v26, i32 0
  %v30 = shl i32 %v27, 2
  store i32 %v30, ptr %v29, align 4
  %v31 = load i32, ptr %v9, align 4
  %v32 = lshr i32 %v31, 2
  %v33 = add i32 %v32, -1
  %v34 = getelementptr inbounds %s.0, ptr %v9, i32 %v33, i32 0
  %v35 = load i32, ptr %v34, align 4
  %v36 = and i32 %v35, 3
  %v37 = or i32 %v36, %v30
  store i32 %v37, ptr %v34, align 4
  %v38 = load i32, ptr %v9, align 4
  %v39 = mul i32 %v25, 32
  %v40 = and i32 %v38, 3
  %v41 = or i32 %v40, %v39
  store i32 %v41, ptr %v9, align 4
  br label %b10

b9:                                               ; preds = %b7
  %v42 = and i32 %v14, -2
  store i32 %v42, ptr %v13, align 4
  br label %b10

b10:                                              ; preds = %b9, %b8
  tail call fastcc void @f3()
  %v43 = getelementptr inbounds %s.0, ptr %v9, i32 1
  %v44 = load i32, ptr %v9, align 4
  %v45 = lshr i32 %v44, 2
  %v46 = mul i32 %v45, 4
  %v47 = add i32 %v46, -4
  %v48 = ptrtoint ptr %v43 to i32
  %v49 = zext i32 %v47 to i64
  %v50 = shl nuw i64 %v49, 32
  %v51 = zext i32 %v48 to i64
  br label %b11

b11:                                              ; preds = %b10, %b6
  %v52 = phi i64 [ 0, %b6 ], [ %v51, %b10 ]
  %v53 = phi i64 [ 0, %b6 ], [ %v50, %b10 ]
  %v54 = or i64 %v53, %v52
  ret i64 %v54
}

declare void @f2(ptr) #0

; Function Attrs: inlinehint nounwind
define internal fastcc void @f3() #1 {
b0:
  store i32 0, ptr @g1, align 4, !tbaa !4
  ret void
}

; Function Attrs: nounwind
define void @f4(ptr nocapture %a0) #0 {
b0:
  %v0 = getelementptr inbounds i32, ptr %a0, i32 -1
  tail call void @f2(ptr @g1) #0
  %v1 = load i32, ptr %v0, align 4
  %v2 = or i32 %v1, 2
  store i32 %v2, ptr %v0, align 4
  tail call fastcc void @f3()
  ret void
}

; Function Attrs: nounwind
define ptr @f5(ptr %a0) #0 {
b0:
  tail call void @f2(ptr @g1) #0
  %v0 = tail call ptr @f0(ptr %a0)
  tail call fastcc void @f3()
  ret ptr %v0
}

; Function Attrs: nounwind
define void @f6(ptr %a0, i32 %a1) #0 {
b0:
  %v0 = getelementptr inbounds %s.0, ptr %a0, i32 7, i32 0
  %v1 = mul i32 %a1, 4
  %v2 = add i32 %v1, -32
  store i32 %v2, ptr %v0, align 4
  %v3 = add i32 %a1, -1
  %v4 = getelementptr inbounds %s.0, ptr %a0, i32 %v3, i32 0
  store i32 1, ptr %v4, align 4
  store i32 0, ptr @g1, align 4, !tbaa !4
  store ptr %a0, ptr @g0, align 4
  ret void
}

; Function Attrs: inlinehint nounwind
define internal fastcc void @f7() #1 {
b0:
  tail call void asm sideeffect " nop", "~{memory}"() #0, !srcloc !6
  ret void
}

; Function Attrs: inlinehint nounwind
define internal fastcc void @f8() #1 {
b0:
  tail call void asm sideeffect " nop", "~{memory}"() #0, !srcloc !7
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { inlinehint nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !2}
!6 = !{i32 782713}
!7 = !{i32 782625}
