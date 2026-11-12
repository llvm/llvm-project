; RUN: llc -mtriple=hexagon-unknown-elf -relocation-model=pic < %s | FileCheck %s

; CHECK: r{{[0-9]+}} = add({{pc|PC}},##.LJTI{{[0-9_]+}}@PCREL)
; CHECK: r{{[0-9]+}} = memw(r{{[0-9]}}+##g0@GOT
; CHECK: r{{[0-9]+}} = add({{pc|PC}},##_GLOBAL_OFFSET_TABLE_@PCREL)
; CHECK: r{{[0-9]+}} = memw(r{{[0-9]*}}+##g1@GOT)

@g0 = external global i32
@g1 = external global i32

; Function Attrs: nounwind
define i32 @f0(i32 %a0) #0 {
b0:
  switch i32 %a0, label %b8 [
    i32 2, label %b1
    i32 3, label %b2
    i32 4, label %b3
    i32 5, label %b4
    i32 6, label %b5
    i32 7, label %b6
    i32 8, label %b7
  ]

b1:                                               ; preds = %b0
  tail call void @f1() #0
  br label %b8

b2:                                               ; preds = %b0
  %v0 = load i32, ptr @g0, align 4, !tbaa !0
  %v1 = add nsw i32 %v0, 99
  br label %b9

b3:                                               ; preds = %b0
  %v2 = load i32, ptr @g1, align 4, !tbaa !0
  %v3 = load i32, ptr @g0, align 4, !tbaa !0
  %v4 = add nsw i32 %v3, %v2
  tail call void @f2(i32 %v4) #0
  br label %b8

b4:                                               ; preds = %b0
  %v5 = load i32, ptr @g1, align 4, !tbaa !0
  %v6 = load i32, ptr @g0, align 4, !tbaa !0
  %v7 = mul nsw i32 %v6, 2
  %v8 = add i32 %v5, 9
  %v9 = add i32 %v8, %v7
  tail call void @f2(i32 %v9) #0
  br label %b8

b5:                                               ; preds = %b0
  br label %b8

b6:                                               ; preds = %b0
  br label %b7

b7:                                               ; preds = %b6, %b0
  %v10 = phi i32 [ 2, %b0 ], [ 4, %b6 ]
  br label %b8

b8:                                               ; preds = %b7, %b5, %b4, %b3, %b1, %b0
  %v11 = phi i32 [ %a0, %b0 ], [ %v10, %b7 ], [ 7, %b5 ], [ 5, %b4 ], [ 4, %b3 ], [ 2, %b1 ]
  %v12 = add nsw i32 %v11, 522
  br label %b9

b9:                                               ; preds = %b8, %b2
  %v13 = phi i32 [ %v12, %b8 ], [ %v1, %b2 ]
  ret i32 %v13
}

declare void @f1(...)

declare void @f2(i32)

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
