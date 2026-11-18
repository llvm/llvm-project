; RUN: llc -march=hexagon -disable-machine-sink=true \
; RUN:     -disable-block-placement=0 -prevent-duplex-separation=false \
; RUN:     -check-early-avail=false -rdf-opt=0 -hexagon-initial-cfg-cleanup=0 \
; RUN:     -hexagon-instsimplify=0 \
; RUN:     < %s | FileCheck %s
; RUN: llc -march=hexagon -disable-machine-sink=true \
; RUN:     -disable-block-placement=0 -rdf-opt=0 \
; RUN:     -hexagon-initial-cfg-cleanup=0 -hexagon-instsimplify=0 \
; RUN:     < %s | FileCheck --check-prefix=CHECK1 %s
; REQUIRES: asserts
; Checks that compound instructions are formed finally (without asserting)
; CHECK-LABEL: f2
; TODO: Enable these when more compound instructions are added.
;  LBB0_2:
;  p{{[0-9]}}=cmp.eq(r{{[0-9]*}},#0); if (p{{[0-9]}}.new) jump:nt .LBB
; CHECK-LABEL: LBB0_21:
; CHECK: p{{[0-9]}} = cmp.eq(r{{[0-9]*}},#0); if (!p{{[0-9]}}.new) jump:nt .LBB


; When duplex has to be preserved.
; CHECK1: %bb.30
; CHECK1: r{{[0-9]+}} = memub(r0+#{{.*}})
; CHECK1: r{{[0-9]+}} = memub(r0+#{{.*}})
; CHECK1: %bb.32
; Without preserving duplexes, both of these instructions are predicated:
;   if (!p0.new) r3 = memub(r0 + #0)
;   if (!p0.new) r2 = memub(r0 + #1)

target triple = "hexagon-unknown--elf"

%0 = type { %1 }
%1 = type { i32, ptr }
%2 = type { i8, i8, i32, i32, i32, i8, %3, i16, i16, i8, ptr, [2 x ptr], i8, %7, [2 x %9], ptr, i8, i16, [256 x i8], [256 x i8], [256 x i8], [20 x i8], [256 x i8], [256 x i8], i16, [256 x i8], i16, %10, %10, %10, [255 x i8], i16, i8, i8, ptr, i32, i8, i32, ptr, i8, %10, %10, ptr, ptr, ptr, ptr, ptr, i32, i16, ptr, i8, ptr, i16, %12, %12, ptr, ptr, [4 x %16], i8, ptr, [8 x i8], [8 x i8], i8, i8, i8 }
%3 = type { i8, %4 }
%4 = type { %5 }
%5 = type { %6 }
%6 = type { [2 x i64] }
%7 = type { [2 x %8], i8 }
%8 = type { i8, i8, i8, [16 x ptr] }
%9 = type { i32, i8 }
%10 = type { i32, %11 }
%11 = type { %5 }
%12 = type { i8, [8 x %13] }
%13 = type { i8, i8, i16, i16, i16, %14, %15 }
%14 = type { %5 }
%15 = type { %5 }
%16 = type { ptr, i32 }

@g0 = external unnamed_addr constant [17 x i8], align 1
@g1 = external constant [29 x %0], section ".......................", align 16

; Function Attrs: optsize
declare void @f0(ptr) #0

; Function Attrs: optsize
declare void @f1(ptr, ptr, i32) #0

; Function Attrs: nounwind optsize
define i32 @f2(ptr %a0, ptr %a1, ptr %a2) #1 {
b0:
  %v0 = alloca i8, align 1
  %v1 = alloca ptr, align 4
  %v2 = alloca i16, align 2
  %v3 = alloca ptr, align 4
  %v4 = alloca i8, align 1
  br i1 undef, label %b2, label %b1

b1:                                               ; preds = %b0
  br i1 undef, label %b2, label %b3

b2:                                               ; preds = %b1, %b0
  unreachable

b3:                                               ; preds = %b1
  %v5 = icmp eq i32 undef, 0
  br i1 %v5, label %b5, label %b4

b4:                                               ; preds = %b3
  unreachable

b5:                                               ; preds = %b3
  %v6 = icmp eq i32 0, 0
  br i1 %v6, label %b7, label %b6

b6:                                               ; preds = %b5
  br label %b64

b7:                                               ; preds = %b5
  br i1 undef, label %b10, label %b8

b8:                                               ; preds = %b7
  br i1 undef, label %b9, label %b12

b9:                                               ; preds = %b8
  unreachable

b10:                                              ; preds = %b7
  br i1 undef, label %b11, label %b12

b11:                                              ; preds = %b10
  unreachable

b12:                                              ; preds = %b10, %b8
  %v7 = icmp eq i8 undef, 0
  br i1 %v7, label %b14, label %b13

b13:                                              ; preds = %b12
  unreachable

b14:                                              ; preds = %b12
  br i1 undef, label %b16, label %b15

b15:                                              ; preds = %b14
  unreachable

b16:                                              ; preds = %b14
  br i1 false, label %b17, label %b18

b17:                                              ; preds = %b16
  br label %b19

b18:                                              ; preds = %b16
  br label %b19

b19:                                              ; preds = %b18, %b17
  %v8 = icmp eq i32 undef, 0
  br i1 %v8, label %b20, label %b21

b20:                                              ; preds = %b19
  %v9 = getelementptr inbounds %2, ptr %a1, i32 0, i32 11, i32 0
  %v10 = getelementptr inbounds i8, ptr null, i32 2006
  %v11 = bitcast ptr %v10 to ptr
  %v12 = getelementptr inbounds %2, ptr %a1, i32 0, i32 53
  br label %b22

b21:                                              ; preds = %b19
  br label %b64

b22:                                              ; preds = %b62, %b20
  %v13 = phi i8 [ 0, %b20 ], [ %v85, %b62 ]
  %v14 = phi i32 [ 0, %b20 ], [ %v84, %b62 ]
  %v15 = phi ptr [ null, %b20 ], [ %v83, %b62 ]
  %v16 = call i32 @f3(ptr null) #1
  %v17 = icmp eq i32 %v16, 0
  br i1 %v17, label %b63, label %b23

b23:                                              ; preds = %b22
  %v18 = load i8, ptr %v0, align 1, !tbaa !0
  %v19 = zext i8 %v18 to i32
  switch i32 %v19, label %b56 [
    i32 46, label %b24
    i32 41, label %b28
    i32 33, label %b45
    i32 40, label %b48
    i32 34, label %b49
    i32 44, label %b52
    i32 45, label %b55
  ]

b24:                                              ; preds = %b23
  %v20 = call i32 @f4(ptr %a0, ptr %a1, ptr %v0, ptr undef) #1
  switch i32 %v20, label %b26 [
    i32 120, label %b25
    i32 0, label %b27
  ]

b25:                                              ; preds = %b24
  unreachable

b26:                                              ; preds = %b24
  call void @f0(ptr undef) #1
  br label %b64

b27:                                              ; preds = %b24
  %v21 = call i32 @f5(ptr null) #1
  br label %b62

b28:                                              ; preds = %b23
  %v22 = call i32 @f10(ptr %a0, ptr %v0, ptr null, ptr %v2, ptr %v3, ptr undef, ptr %v1) #1
  %v23 = icmp eq i32 %v22, 0
  br i1 %v23, label %b30, label %b29

b29:                                              ; preds = %b28
  call void @f0(ptr null) #1
  store i16 7, ptr %v11, align 2, !tbaa !3
  br label %b64

b30:                                              ; preds = %b28
  %v24 = load i16, ptr %v2, align 2, !tbaa !16
  %v25 = call i32 @f11(ptr %a1, i16 zeroext %v24, ptr null) #1
  %v26 = icmp eq i32 %v25, 0
  br i1 %v26, label %b32, label %b31

b31:                                              ; preds = %b30
  call void @f0(ptr null) #1
  unreachable

b32:                                              ; preds = %b30
  %v27 = load i16, ptr %v2, align 2, !tbaa !16
  %v28 = icmp eq i16 %v27, 16391
  %v29 = select i1 %v28, i8 1, i8 %v13
  %v30 = icmp eq i16 %v27, 16393
  %v31 = load ptr, ptr %v3, align 4, !tbaa !17
  br i1 %v30, label %b33, label %b42

b33:                                              ; preds = %b32
  %v32 = icmp eq ptr %v31, null
  br i1 %v32, label %b34, label %b35

b34:                                              ; preds = %b33
  call void @f0(ptr null) #1
  br label %b64

b35:                                              ; preds = %b33
  %v33 = load i8, ptr %v31, align 1, !tbaa !0
  %v34 = zext i8 %v33 to i32
  %v35 = mul i32 %v34, 16777216
  %v36 = getelementptr inbounds i8, ptr %v31, i32 1
  %v37 = load i8, ptr %v36, align 1, !tbaa !0
  %v38 = zext i8 %v37 to i32
  %v39 = mul i32 %v38, 65536
  %v40 = add nsw i32 %v39, %v35
  %v41 = load i8, ptr undef, align 1, !tbaa !0
  %v42 = zext i8 %v41 to i32
  %v43 = mul i32 %v42, 256
  %v44 = add nsw i32 %v40, %v43
  %v45 = getelementptr inbounds i8, ptr %v31, i32 3
  %v46 = load i8, ptr %v45, align 1, !tbaa !0
  %v47 = zext i8 %v46 to i32
  %v48 = add nsw i32 %v44, %v47
  br label %b36

b36:                                              ; preds = %b39, %b35
  %v49 = phi i32 [ 0, %b35 ], [ %v57, %b39 ]
  %v50 = getelementptr inbounds [16 x ptr], ptr null, i32 0, i32 %v49
  %v51 = load ptr, ptr %v50, align 4, !tbaa !17
  br i1 undef, label %b39, label %b37

b37:                                              ; preds = %b36
  %v52 = getelementptr inbounds i8, ptr %v51, i32 32
  %v53 = bitcast ptr %v52 to ptr
  %v54 = load i32, ptr %v53, align 4, !tbaa !18
  %v55 = icmp eq i32 %v54, %v48
  br i1 %v55, label %b38, label %b39

b38:                                              ; preds = %b37
  store i8 1, ptr null, align 1, !tbaa !24
  %v56 = load ptr, ptr %v50, align 4, !tbaa !17
  store ptr %v56, ptr %v9, align 4, !tbaa !17
  store i8 1, ptr undef, align 1, !tbaa !25
  br label %b40

b39:                                              ; preds = %b37, %b36
  %v57 = add i32 %v49, 1
  %v58 = icmp ult i32 %v57, 16
  br i1 %v58, label %b36, label %b40

b40:                                              ; preds = %b39, %b38
  %v59 = phi ptr [ undef, %b38 ], [ %v15, %b39 ]
  %v60 = icmp eq ptr %v59, null
  br i1 %v60, label %b41, label %b42

b41:                                              ; preds = %b40
  call void @f0(ptr null) #1
  unreachable

b42:                                              ; preds = %b40, %b32
  %v61 = phi ptr [ %v59, %b40 ], [ %v15, %b32 ]
  %v62 = or i32 %v14, 256
  %v63 = icmp eq ptr %v31, null
  br i1 %v63, label %b44, label %b43

b43:                                              ; preds = %b42
  call void @f12(ptr %v31) #1
  unreachable

b44:                                              ; preds = %b42
  call void @f1(ptr %v1, ptr @g0, i32 3626) #1
  br label %b62

b45:                                              ; preds = %b48, %b23
  %v64 = phi ptr [ @f7, %b48 ], [ @f6, %b23 ]
  %v65 = phi i32 [ 128, %b48 ], [ 1, %b23 ]
  %v66 = phi ptr [ getelementptr inbounds ([29 x %0], ptr @g1, i32 0, i32 17), %b48 ], [ getelementptr inbounds ([29 x %0], ptr @g1, i32 0, i32 16), %b23 ]
  %v67 = call i32 %v64(ptr %a0, ptr %a1, ptr %v0) #1
  %v68 = icmp eq i32 %v67, 0
  br i1 %v68, label %b47, label %b46

b46:                                              ; preds = %b45
  call void @f0(ptr %v66) #1
  store i16 7, ptr %v11, align 2, !tbaa !3
  br label %b64

b47:                                              ; preds = %b45
  %v69 = or i32 %v14, %v65
  br label %b62

b48:                                              ; preds = %b23
  br label %b45

b49:                                              ; preds = %b23
  %v70 = call i32 @f8(ptr %a0, ptr %a1, ptr %v0) #1
  %v71 = icmp eq i32 %v70, 0
  br i1 %v71, label %b51, label %b50

b50:                                              ; preds = %b49
  call void @f0(ptr null) #1
  br label %b64

b51:                                              ; preds = %b49
  %v72 = or i32 %v14, 2
  br label %b62

b52:                                              ; preds = %b55, %b23
  %v73 = phi ptr [ null, %b55 ], [ %v12, %b23 ]
  %v74 = phi i32 [ 4096, %b55 ], [ 2048, %b23 ]
  %v75 = phi ptr [ getelementptr inbounds ([29 x %0], ptr @g1, i32 0, i32 20), %b55 ], [ getelementptr inbounds ([29 x %0], ptr @g1, i32 0, i32 19), %b23 ]
  %v76 = call i32 @f9(ptr %a0, ptr %v0, ptr %v73) #1
  %v77 = icmp eq i32 %v76, 0
  br i1 %v77, label %b54, label %b53

b53:                                              ; preds = %b52
  call void @f0(ptr %v75) #1
  store i16 7, ptr %v11, align 2, !tbaa !3
  br label %b64

b54:                                              ; preds = %b52
  %v78 = or i32 %v14, %v74
  br label %b62

b55:                                              ; preds = %b23
  br label %b52

b56:                                              ; preds = %b23
  %v79 = call i32 @f13(ptr %a0, ptr %v0, ptr %v4) #1
  %v80 = icmp eq i32 %v79, 0
  br i1 %v80, label %b58, label %b57

b57:                                              ; preds = %b56
  call void @f0(ptr undef) #1
  unreachable

b58:                                              ; preds = %b56
  %v81 = load i8, ptr %v4, align 1, !tbaa !0
  %v82 = icmp eq i8 %v81, 0
  br i1 %v82, label %b62, label %b59

b59:                                              ; preds = %b58
  br i1 undef, label %b60, label %b61

b60:                                              ; preds = %b59
  unreachable

b61:                                              ; preds = %b59
  store i16 1, ptr %v11, align 2, !tbaa !3
  store i8 %v18, ptr undef, align 1, !tbaa !0
  br label %b64

b62:                                              ; preds = %b58, %b54, %b51, %b47, %b44, %b27
  %v83 = phi ptr [ %v15, %b58 ], [ %v15, %b54 ], [ %v15, %b51 ], [ %v15, %b47 ], [ %v15, %b27 ], [ %v61, %b44 ]
  %v84 = phi i32 [ %v14, %b58 ], [ %v78, %b54 ], [ %v72, %b51 ], [ %v69, %b47 ], [ undef, %b27 ], [ %v62, %b44 ]
  %v85 = phi i8 [ %v13, %b58 ], [ %v13, %b54 ], [ %v13, %b51 ], [ %v13, %b47 ], [ %v13, %b27 ], [ %v29, %b44 ]
  %v86 = load i8, ptr %v0, align 1, !tbaa !0
  %v87 = icmp eq i8 %v86, 0
  br i1 %v87, label %b63, label %b22

b63:                                              ; preds = %b62, %b22
  call void @f0(ptr getelementptr inbounds ([29 x %0], ptr @g1, i32 0, i32 26)) #1
  unreachable

b64:                                              ; preds = %b61, %b53, %b50, %b46, %b34, %b29, %b26, %b21, %b6
  %v88 = phi i32 [ 1, %b6 ], [ undef, %b21 ], [ 1, %b61 ], [ %v76, %b53 ], [ %v67, %b46 ], [ %v22, %b29 ], [ 1, %b34 ], [ %v20, %b26 ], [ %v70, %b50 ]
  br i1 undef, label %b66, label %b65

b65:                                              ; preds = %b64
  unreachable

b66:                                              ; preds = %b64
  ret i32 %v88
}

; Function Attrs: optsize
declare i32 @f3(ptr) #0

; Function Attrs: optsize
declare i32 @f4(ptr, ptr, ptr, ptr) #0

; Function Attrs: optsize
declare i32 @f5(ptr) #0

; Function Attrs: optsize
declare i32 @f6(ptr, ptr, ptr) #0

; Function Attrs: optsize
declare i32 @f7(ptr, ptr, ptr) #0

; Function Attrs: optsize
declare i32 @f8(ptr, ptr, ptr) #0

; Function Attrs: optsize
declare i32 @f9(ptr, ptr, ptr) #0

; Function Attrs: optsize
declare i32 @f10(ptr, ptr, ptr, ptr, ptr, ptr, ptr) #0

; Function Attrs: optsize
declare i32 @f11(ptr, i16 zeroext, ptr) #0

; Function Attrs: nounwind optsize
declare void @f12(ptr nocapture) #1

; Function Attrs: optsize
declare i32 @f13(ptr, ptr, ptr) #0

attributes #0 = { optsize }
attributes #1 = { nounwind optsize }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!4, !1, i64 2006}
!4 = !{!"", !1, i64 0, !5, i64 4, !5, i64 8, !5, i64 12, !5, i64 16, !5, i64 20, !5, i64 24, !6, i64 28, !7, i64 40, !7, i64 64, !8, i64 88, !1, i64 92, !1, i64 93, !1, i64 96, !8, i64 100, !1, i64 104, !8, i64 108, !1, i64 112, !8, i64 116, !1, i64 120, !9, i64 124, !5, i64 196, !5, i64 200, !1, i64 204, !1, i64 236, !1, i64 256, !1, i64 276, !10, i64 292, !10, i64 294, !10, i64 296, !10, i64 298, !1, i64 300, !1, i64 301, !8, i64 304, !8, i64 308, !1, i64 312, !5, i64 376, !1, i64 380, !1, i64 381, !1, i64 382, !1, i64 383, !1, i64 384, !11, i64 388, !12, i64 504, !5, i64 1120, !1, i64 1124, !1, i64 1132, !1, i64 1140, !1, i64 1172, !1, i64 1192, !1, i64 1212, !1, i64 1244, !1, i64 1276, !1, i64 1296, !8, i64 1316, !8, i64 1320, !8, i64 1324, !1, i64 1328, !10, i64 1584, !1, i64 1586, !10, i64 1842, !5, i64 1844, !1, i64 1848, !5, i64 1904, !8, i64 1908, !8, i64 1912, !8, i64 1916, !8, i64 1920, !8, i64 1924, !1, i64 1928, !1, i64 1929, !1, i64 1930, !1, i64 1932, !1, i64 1996, !1, i64 1998, !8, i64 2000, !10, i64 2004, !1, i64 2006, !8, i64 2008, !10, i64 2012, !1, i64 2014, !1, i64 2015, !5, i64 2016, !8, i64 2020, !8, i64 2024, !1, i64 2028, !13, i64 2032, !15, i64 2056, !15, i64 2064}
!5 = !{!"long", !1, i64 0}
!6 = !{!"", !5, i64 0, !5, i64 4}
!7 = !{!"", !1, i64 0, !1, i64 8}
!8 = !{!"any pointer", !1, i64 0}
!9 = !{!"", !10, i64 0, !1, i64 4, !10, i64 52, !5, i64 56, !5, i64 60, !1, i64 64, !5, i64 68}
!10 = !{!"short", !1, i64 0}
!11 = !{!"", !1, i64 0, !1, i64 8, !1, i64 16, !1, i64 24, !1, i64 32, !1, i64 33, !5, i64 36, !6, i64 40, !5, i64 48, !5, i64 52, !1, i64 56, !1, i64 57, !5, i64 60, !5, i64 64, !1, i64 68, !1, i64 70, !1, i64 86, !1, i64 87, !1, i64 88, !5, i64 92, !5, i64 96, !5, i64 100, !5, i64 104, !5, i64 108}
!12 = !{!"", !13, i64 0, !13, i64 24, !13, i64 48, !13, i64 72, !14, i64 96, !14, i64 356}
!13 = !{!"", !5, i64 0, !1, i64 8}
!14 = !{!"", !1, i64 0, !1, i64 2}
!15 = !{!"", !1, i64 0, !8, i64 4}
!16 = !{!10, !10, i64 0}
!17 = !{!8, !8, i64 0}
!18 = !{!19, !5, i64 32}
!19 = !{!"", !20, i64 0, !21, i64 8, !21, i64 32, !7, i64 56, !7, i64 80, !1, i64 104, !1, i64 105, !5, i64 108, !5, i64 112, !1, i64 116, !1, i64 117, !9, i64 120, !1, i64 192, !8, i64 196, !22, i64 200, !23, i64 244, !13, i64 256, !13, i64 280, !5, i64 304, !5, i64 308, !5, i64 312, !10, i64 316, !10, i64 318, !10, i64 320, !10, i64 322, !8, i64 324, !8, i64 328, !1, i64 332, !1, i64 348, !1, i64 368, !1, i64 388, !1, i64 420, !1, i64 452, !7, i64 456, !7, i64 480, !8, i64 504, !8, i64 508, !8, i64 512, !8, i64 516, !8, i64 520, !8, i64 524, !8, i64 528, !1, i64 532, !1, i64 536}
!20 = !{!"q_link_struct", !8, i64 0, !8, i64 4}
!21 = !{!"", !5, i64 0, !5, i64 4, !5, i64 8, !5, i64 12, !6, i64 16}
!22 = !{!"", !8, i64 0, !1, i64 4, !1, i64 12, !1, i64 20, !1, i64 28}
!23 = !{!"", !8, i64 0, !1, i64 4}
!24 = !{!4, !1, i64 1928}
!25 = !{!26, !1, i64 68}
!26 = !{!"", !1, i64 0, !1, i64 1, !5, i64 4, !5, i64 8, !5, i64 12, !1, i64 16, !7, i64 24, !10, i64 48, !10, i64 50, !1, i64 52, !8, i64 56, !1, i64 60, !1, i64 68, !27, i64 72, !1, i64 212, !8, i64 228, !1, i64 232, !10, i64 234, !1, i64 236, !1, i64 492, !1, i64 748, !1, i64 1004, !1, i64 1024, !1, i64 1280, !10, i64 1536, !1, i64 1538, !10, i64 1794, !13, i64 1800, !13, i64 1824, !13, i64 1848, !1, i64 1872, !10, i64 2128, !1, i64 2130, !1, i64 2131, !8, i64 2132, !1, i64 2136, !1, i64 2140, !1, i64 2144, !8, i64 2148, !1, i64 2152, !13, i64 2160, !13, i64 2184, !8, i64 2208, !8, i64 2212, !8, i64 2216, !8, i64 2220, !8, i64 2224, !5, i64 2228, !10, i64 2232, !8, i64 2236, !1, i64 2240, !8, i64 2244, !10, i64 2248, !7, i64 2256, !7, i64 2584, !8, i64 2912, !8, i64 2916, !1, i64 2920, !1, i64 2952, !8, i64 2956, !1, i64 2960, !1, i64 2968, !1, i64 2976, !1, i64 2977, !1, i64 2978}
!27 = !{!"", !1, i64 0, !1, i64 136}
