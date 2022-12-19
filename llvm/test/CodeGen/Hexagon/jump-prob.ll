; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK: {
; CHECK: jump .LBB0_
; CHECK: r{{[0-9]+}} =
; CHECK: memw
; CHECK: }

target triple = "hexagon-unknown--elf"

%s.0 = type { i8, i8, i8, [6 x i32] }
%s.1 = type { %s.2 }
%s.2 = type { i32, ptr }
%s.3 = type <{ ptr, ptr, i16, i8, i8, i8 }>

@g0 = internal global [2 x %s.0] [%s.0 { i8 0, i8 6, i8 7, [6 x i32] zeroinitializer }, %s.0 { i8 0, i8 6, i8 7, [6 x i32] zeroinitializer }], align 8
@g1 = internal constant [60 x i8] c"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\00", section "xxxxxxxxxxx.rodata.", align 4
@g2 = internal constant %s.1 { %s.2 { i32 24, ptr @g1 } }, section ".rodata.xxxxxxxxxx.", align 4
@g3 = internal constant [115 x i8] c"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\00", section "xxxxxxxxxxx.rodata.", align 4
@g4 = internal constant %s.3 <{ ptr @g5, ptr @g6, i16 215, i8 4, i8 0, i8 1 }>, align 1
@g5 = private unnamed_addr constant [120 x i8] c"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\00", align 1
@g6 = private unnamed_addr constant [31 x i8] c"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\00", align 1
@g7 = internal constant %s.3 <{ ptr @g5, ptr @g8, i16 225, i8 2, i8 2, i8 2 }>, align 1
@g8 = private unnamed_addr constant [91 x i8] c"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\00", align 1
@g9 = internal constant %s.3 <{ ptr @g5, ptr @g10, i16 233, i8 2, i8 2, i8 4 }>, align 1
@g10 = private unnamed_addr constant [109 x i8] c"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\00", align 1
@g11 = internal constant [116 x i8] c"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\00", section "xxxxxxxxxxx.rodata.", align 4
@g12 = internal constant [134 x i8] c"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\00", section "xxxxxxxxxxx.rodata.", align 4
@g13 = internal constant %s.3 <{ ptr @g5, ptr @g6, i16 264, i8 4, i8 0, i8 1 }>, align 1
@g14 = internal constant [116 x i8] c"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\00", section "xxxxxxxxxxx.rodata.", align 4
@g15 = internal constant [134 x i8] c"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\00", section "xxxxxxxxxxx.rodata.", align 4

; Function Attrs: nounwind
define zeroext i8 @f0(i8 zeroext %a0, i8 zeroext %a1, ptr nocapture %a2) #0 {
b0:
  store i8 -1, ptr %a2, align 1, !tbaa !0
  %v0 = zext i8 %a0 to i32
  %v1 = icmp ugt i8 %a0, 7
  %v2 = zext i8 %a1 to i32
  %v3 = icmp ugt i8 %a1, 5
  %v4 = or i1 %v1, %v3
  br i1 %v4, label %b1, label %b2

b1:                                               ; preds = %b0
  tail call void @f1(ptr @g2, i32 2, i32 %v0, i32 %v2)
  br label %b12

b2:                                               ; preds = %b0
  %v5 = load i8, ptr getelementptr inbounds ([2 x %s.0], ptr @g0, i32 0, i32 0, i32 2), align 2, !tbaa !0
  %v6 = icmp eq i8 %v5, %a0
  %v7 = load i8, ptr getelementptr inbounds ([2 x %s.0], ptr @g0, i32 0, i32 1, i32 2), align 2, !tbaa !0
  %v8 = icmp eq i8 %v7, %a0
  %v9 = and i1 %v6, %v8
  br i1 %v9, label %b3, label %b4

b3:                                               ; preds = %b2
  %v10 = getelementptr inbounds [2 x %s.0], ptr @g0, i32 0, i32 0, i32 3, i32 %v2
  %v11 = load i32, ptr %v10, align 4, !tbaa !3
  %v12 = getelementptr inbounds [2 x %s.0], ptr @g0, i32 0, i32 1, i32 3, i32 %v2
  %v13 = load i32, ptr %v12, align 4, !tbaa !3
  tail call void @f1(ptr @g2, i32 2, i32 %v0, i32 %v2)
  br label %b12

b4:                                               ; preds = %b2
  %v14 = load i8, ptr @g0, align 8, !tbaa !0
  %v15 = icmp eq i8 %v14, 1
  %v16 = and i1 %v15, %v6
  br i1 %v16, label %b5, label %b8

b5:                                               ; preds = %b4
  store i8 0, ptr %a2, align 1, !tbaa !0
  %v17 = getelementptr inbounds [2 x %s.0], ptr @g0, i32 0, i32 0, i32 3, i32 %v2
  %v18 = tail call i32 asm sideeffect "1:     $0 = memw_locked($2)\0A       $0 = add($0, $3)\0A       memw_locked($2, p0) = $0\0A       if !p0 jump 1b\0A", "=&r,=*m,r,r,*m,~{p0}"(ptr elementtype(i32) %v17, ptr %v17, i32 1, ptr elementtype(i32) %v17) #0, !srcloc !5
  %v19 = load i32, ptr %v17, align 4, !tbaa !3
  %v20 = icmp eq i32 %v19, 255
  br i1 %v20, label %b6, label %b7

b6:                                               ; preds = %b5
  tail call void @f2(ptr @g4, i32 %v2) #2
  unreachable

b7:                                               ; preds = %b5
  store i8 %a1, ptr getelementptr inbounds ([2 x %s.0], ptr @g0, i32 0, i32 0, i32 1), align 1, !tbaa !0
  %v21 = load i8, ptr %a2, align 1, !tbaa !0
  %v22 = zext i8 %v21 to i32
  tail call void @f3(ptr @g7, i32 %v2, i32 %v22) #0
  %v23 = load i32, ptr @g0, align 8
  %v24 = and i32 %v23, 255
  %v25 = lshr i32 %v23, 8
  %v26 = and i32 %v25, 255
  %v27 = lshr i32 %v23, 16
  %v28 = and i32 %v27, 255
  %v29 = load i32, ptr %v17, align 4, !tbaa !3
  tail call void @f4(ptr @g9, i32 %v24, i32 %v26, i32 %v28, i32 %v29) #0
  %v30 = load i8, ptr %a2, align 1, !tbaa !0
  %v31 = zext i8 %v30 to i32
  tail call void @f1(ptr @g2, i32 2, i32 %v0, i32 %v2)
  %v32 = load i32, ptr @g0, align 8
  %v33 = and i32 %v32, 255
  %v34 = lshr i32 %v32, 8
  %v35 = and i32 %v34, 255
  %v36 = lshr i32 %v32, 16
  %v37 = and i32 %v36, 255
  %v38 = load i32, ptr %v17, align 4, !tbaa !3
  tail call void @f1(ptr @g2, i32 2, i32 %v0, i32 %v2)
  br label %b12

b8:                                               ; preds = %b4
  %v39 = load i8, ptr getelementptr inbounds ([2 x %s.0], ptr @g0, i32 0, i32 1, i32 0), align 4, !tbaa !0
  %v40 = icmp eq i8 %v39, 1
  %v41 = and i1 %v40, %v8
  br i1 %v41, label %b9, label %b12

b9:                                               ; preds = %b8
  store i8 1, ptr %a2, align 1, !tbaa !0
  %v42 = getelementptr inbounds [2 x %s.0], ptr @g0, i32 0, i32 1, i32 3, i32 %v2
  %v43 = tail call i32 asm sideeffect "1:     $0 = memw_locked($2)\0A       $0 = add($0, $3)\0A       memw_locked($2, p0) = $0\0A       if !p0 jump 1b\0A", "=&r,=*m,r,r,*m,~{p0}"(ptr elementtype(i32) %v42, ptr %v42, i32 1, ptr elementtype(i32) %v42) #0, !srcloc !5
  %v44 = load i32, ptr %v42, align 4, !tbaa !3
  %v45 = icmp eq i32 %v44, 255
  br i1 %v45, label %b10, label %b11

b10:                                              ; preds = %b9
  tail call void @f2(ptr @g13, i32 %v2) #2
  unreachable

b11:                                              ; preds = %b9
  store i8 %a1, ptr getelementptr inbounds ([2 x %s.0], ptr @g0, i32 0, i32 1, i32 1), align 1, !tbaa !0
  %v46 = load i8, ptr %a2, align 1, !tbaa !0
  %v47 = zext i8 %v46 to i32
  tail call void @f1(ptr @g2, i32 2, i32 %v0, i32 %v2)
  %v48 = load i32, ptr getelementptr inbounds ([2 x %s.0], ptr @g0, i32 0, i32 1, i32 0), align 4
  %v49 = and i32 %v48, 255
  %v50 = lshr i32 %v48, 8
  %v51 = and i32 %v50, 255
  %v52 = lshr i32 %v48, 16
  %v53 = and i32 %v52, 255
  %v54 = load i32, ptr %v42, align 4, !tbaa !3
  tail call void @f1(ptr @g2, i32 2, i32 %v0, i32 %v2)
  br label %b12

b12:                                              ; preds = %b11, %b8, %b7, %b3, %b1
  %v55 = phi i8 [ 0, %b1 ], [ 0, %b3 ], [ 1, %b7 ], [ 1, %b11 ], [ 0, %b8 ]
  ret i8 %v55
}

declare void @f1(ptr, i32, i32, i32)

; Function Attrs: noreturn
declare void @f2(ptr, i32) #1

declare void @f3(ptr, i32, i32)

declare void @f4(ptr, i32, i32, i32, i32)

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { noreturn }
attributes #2 = { noreturn nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!4, !4, i64 0}
!4 = !{!"long", !1}
!5 = !{i32 86170, i32 86211, i32 86247, i32 86291}
