; RUN: llc -march=hexagon -mcpu=hexagonv73 -O3 -prevent-duplex-sep-v2=false \
; RUN:   -hexagon-initial-cfg-cleanup=0 -disable-nvjump < %s | FileCheck %s

; Disable new-value jumps so the compare keeps a predicate register that the
; speculatively pulled-up load can use.

; CHECK: memw(##g0)
; CHECK: addasl
; CHECK: if (!p{{[0-2]}}.new) r{{[0-9]*}} = memw

target datalayout = "e-m:e-p:32:32-i1:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon-unknown--elf"

%s.3 = type { i32, i32, i32, ptr, i32, [10 x %s.2] }
%s.2 = type { ptr, i8 }

@g0 = internal unnamed_addr global i32 0, align 4
@g1 = internal unnamed_addr global i32 0, align 4

; Function Attrs: nounwind
define fastcc void @f0(ptr %a0, ptr %a1, ptr nocapture readonly %a2, i32 %a3) #0 {
b0:
  %v0 = alloca %s.3, align 4
  br label %b1

b1:                                               ; preds = %b1, %b0
  br i1 poison, label %b2, label %b1

b2:                                               ; preds = %b1
  %v1 = icmp eq ptr %a1, null
  br i1 %v1, label %b5, label %b3

b3:                                               ; preds = %b2
  br i1 poison, label %b4, label %b5

b4:                                               ; preds = %b3
  store i32 poison, ptr null, align 4, !tbaa !0
  br label %b5

b5:                                               ; preds = %b4, %b3, %b2
  br i1 poison, label %b40, label %b6

b6:                                               ; preds = %b5
  %v2 = getelementptr inbounds %s.3, ptr %v0, i32 0, i32 2
  %v3 = getelementptr inbounds %s.3, ptr %v0, i32 0, i32 3
  br label %b7

b7:                                               ; preds = %b39, %b6
  br i1 poison, label %b8, label %b18

b8:                                               ; preds = %b7
  switch i8 0, label %b10 [
    i8 1, label %b18
    i8 0, label %b9
  ]

b9:                                               ; preds = %b8
  br label %b17

b10:                                              ; preds = %b8
  br i1 false, label %b17, label %b11

b11:                                              ; preds = %b16, %b10
  br i1 poison, label %b17, label %b12

b12:                                              ; preds = %b11
  br i1 false, label %b13, label %b16

b13:                                              ; preds = %b12
  br i1 poison, label %b14, label %b15

b14:                                              ; preds = %b13
  br label %b15

b15:                                              ; preds = %b14, %b13
  br label %b16

b16:                                              ; preds = %b15, %b12
  br i1 false, label %b17, label %b11

b17:                                              ; preds = %b16, %b11, %b10, %b9
  br label %b39

b18:                                              ; preds = %b8, %b7
  br i1 poison, label %b19, label %b20

b19:                                              ; preds = %b18
  br label %b20

b20:                                              ; preds = %b19, %b18
  br i1 poison, label %b21, label %b26

b21:                                              ; preds = %b20
  br i1 poison, label %b23, label %b22

b22:                                              ; preds = %b21
  unreachable

b23:                                              ; preds = %b21
  br i1 poison, label %b24, label %b25

b24:                                              ; preds = %b23
  br label %b25

b25:                                              ; preds = %b24, %b23
  br label %b26

b26:                                              ; preds = %b25, %b20
  br i1 poison, label %b30, label %b27

b27:                                              ; preds = %b26
  br i1 poison, label %b63, label %b28

b28:                                              ; preds = %b27
  %v8 = icmp eq i16 0, 0
  br i1 %v8, label %b29, label %b40

b29:                                              ; preds = %b28
  br label %b30

b30:                                              ; preds = %b29, %b26
  br i1 poison, label %b39, label %b31

b31:                                              ; preds = %b30
  %v9 = icmp eq ptr poison, null
  br i1 %v9, label %b32, label %b39

b32:                                              ; preds = %b31
  %v10 = load i16, ptr poison, align 2, !tbaa !4
  %v11 = icmp eq i16 %v10, 31
  br i1 %v11, label %b33, label %b39

b33:                                              ; preds = %b32
  br i1 poison, label %b34, label %b39

b34:                                              ; preds = %b33
  br i1 false, label %b35, label %b36

b35:                                              ; preds = %b34
  unreachable

b36:                                              ; preds = %b34
  br i1 false, label %b63, label %b37

b37:                                              ; preds = %b36
  %v15 = load i32, ptr %v2, align 4, !tbaa !7
  %v16 = mul nsw i32 %v15, 2
  %v17 = load i32, ptr @g1, align 4, !tbaa !0
  %v18 = add nsw i32 %v16, %v17
  %v19 = load i32, ptr @g0, align 4, !tbaa !0
  %v20 = icmp sgt i32 %v18, %v19
  br i1 %v20, label %b40, label %b38

b38:                                              ; preds = %b37
  %v21 = load ptr, ptr %v3, align 4, !tbaa !10
  br label %b39

b39:                                              ; preds = %b38, %b33, %b32, %b31, %b30, %b17
  %v22 = phi ptr [ null, %b17 ], [ null, %b33 ], [ null, %b32 ], [ null, %b31 ], [ null, %b30 ], [ %v21, %b38 ]
  %v25 = icmp eq ptr null, %v22
  br i1 %v25, label %b40, label %b7

b40:                                              ; preds = %b39, %b37, %b28, %b5
  %v26 = phi ptr [ %a0, %b5 ], [ null, %b37 ], [ null, %b28 ], [ %v22, %b39 ]
  br i1 poison, label %b41, label %b42

b41:                                              ; preds = %b40
  unreachable

b42:                                              ; preds = %b40
  br i1 poison, label %b44, label %b43

b43:                                              ; preds = %b42
  br i1 poison, label %b45, label %b63

b44:                                              ; preds = %b42
  %v27 = icmp ne i32 %a3, 0
  %v28 = icmp ne ptr %v26, null
  %v29 = and i1 %v27, %v28
  br i1 %v29, label %b45, label %b63

b45:                                              ; preds = %b44, %b43
  %v30 = load i16, ptr poison, align 2, !tbaa !4
  %v31 = icmp eq i16 %v30, 32
  br i1 %v31, label %b46, label %b63

b46:                                              ; preds = %b45
  br i1 poison, label %b47, label %b63

b47:                                              ; preds = %b46
  br i1 poison, label %b48, label %b63

b48:                                              ; preds = %b47
  br i1 poison, label %b63, label %b49

b49:                                              ; preds = %b48
  br i1 poison, label %b50, label %b63

b50:                                              ; preds = %b52, %b49
  br i1 false, label %b63, label %b51

b51:                                              ; preds = %b50
  br i1 poison, label %b52, label %b63

b52:                                              ; preds = %b51
  br i1 poison, label %b50, label %b53

b53:                                              ; preds = %b52
  br i1 false, label %b54, label %b63

b54:                                              ; preds = %b53
  br i1 poison, label %b59, label %b55

b55:                                              ; preds = %b54
  br i1 poison, label %b59, label %b56

b56:                                              ; preds = %b55
  switch i16 poison, label %b57 [
    i16 113, label %b58
    i16 55, label %b58
  ]

b57:                                              ; preds = %b56
  unreachable

b58:                                              ; preds = %b56, %b56
  unreachable

b59:                                              ; preds = %b61, %b60, %b55, %b54
  switch i16 poison, label %b60 [
    i16 29, label %b63
    i16 31, label %b63
  ]

b60:                                              ; preds = %b59
  br i1 poison, label %b61, label %b59

b61:                                              ; preds = %b60
  switch i16 poison, label %b59 [
    i16 41, label %b62
    i16 43, label %b62
  ]

b62:                                              ; preds = %b61, %b61
  unreachable

b63:                                              ; preds = %b59, %b59, %b53, %b51, %b50, %b49, %b48, %b47, %b46, %b45, %b44, %b43, %b36, %b27
  ret void
}

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !6, i64 0}
!5 = !{!"rtx_def", !6, i64 0, !1, i64 2, !1, i64 3, !1, i64 3, !1, i64 3, !1, i64 3, !1, i64 3, !1, i64 3, !1, i64 3, !2, i64 4}
!6 = !{!"short", !2, i64 0}
!7 = !{!8, !1, i64 8}
!8 = !{!"cse_basic_block_data", !1, i64 0, !1, i64 4, !1, i64 8, !9, i64 12, !1, i64 16, !2, i64 20}
!9 = !{!"any pointer", !2, i64 0}
!10 = !{!8, !9, i64 12}
