; RUN: llc -march=hexagon -relocation-model=pic -O2 < %s | FileCheck %s
; CHECK: r{{[0-9]+}} = add(pc,##.Ltmp0@PCREL)
; CHECK-NOT: r{{[0-9]+}} = ##.Ltmp0

target triple = "hexagon"

%s.0 = type { [7 x ptr], [7 x ptr], [12 x ptr], [12 x ptr], [2 x ptr], ptr, ptr, ptr, ptr }
%s.1 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32 }

@g0 = private unnamed_addr constant [4 x i8] c"Sun\00", align 1
@g1 = private unnamed_addr constant [4 x i8] c"Mon\00", align 1
@g2 = private unnamed_addr constant [4 x i8] c"Tue\00", align 1
@g3 = private unnamed_addr constant [4 x i8] c"Wed\00", align 1
@g4 = private unnamed_addr constant [4 x i8] c"Thu\00", align 1
@g5 = private unnamed_addr constant [4 x i8] c"Fri\00", align 1
@g6 = private unnamed_addr constant [4 x i8] c"Sat\00", align 1
@g7 = private unnamed_addr constant [7 x i8] c"Sunday\00", align 1
@g8 = private unnamed_addr constant [7 x i8] c"Monday\00", align 1
@g9 = private unnamed_addr constant [8 x i8] c"Tuesday\00", align 1
@g10 = private unnamed_addr constant [10 x i8] c"Wednesday\00", align 1
@g11 = private unnamed_addr constant [9 x i8] c"Thursday\00", align 1
@g12 = private unnamed_addr constant [7 x i8] c"Friday\00", align 1
@g13 = private unnamed_addr constant [9 x i8] c"Saturday\00", align 1
@g14 = private unnamed_addr constant [4 x i8] c"Jan\00", align 1
@g15 = private unnamed_addr constant [4 x i8] c"Feb\00", align 1
@g16 = private unnamed_addr constant [4 x i8] c"Mar\00", align 1
@g17 = private unnamed_addr constant [4 x i8] c"Apr\00", align 1
@g18 = private unnamed_addr constant [4 x i8] c"May\00", align 1
@g19 = private unnamed_addr constant [4 x i8] c"Jun\00", align 1
@g20 = private unnamed_addr constant [4 x i8] c"Jul\00", align 1
@g21 = private unnamed_addr constant [4 x i8] c"Aug\00", align 1
@g22 = private unnamed_addr constant [4 x i8] c"Sep\00", align 1
@g23 = private unnamed_addr constant [4 x i8] c"Oct\00", align 1
@g24 = private unnamed_addr constant [4 x i8] c"Nov\00", align 1
@g25 = private unnamed_addr constant [4 x i8] c"Dec\00", align 1
@g26 = private unnamed_addr constant [8 x i8] c"January\00", align 1
@g27 = private unnamed_addr constant [9 x i8] c"February\00", align 1
@g28 = private unnamed_addr constant [6 x i8] c"March\00", align 1
@g29 = private unnamed_addr constant [6 x i8] c"April\00", align 1
@g30 = private unnamed_addr constant [5 x i8] c"June\00", align 1
@g31 = private unnamed_addr constant [5 x i8] c"July\00", align 1
@g32 = private unnamed_addr constant [7 x i8] c"August\00", align 1
@g33 = private unnamed_addr constant [10 x i8] c"September\00", align 1
@g34 = private unnamed_addr constant [8 x i8] c"October\00", align 1
@g35 = private unnamed_addr constant [9 x i8] c"November\00", align 1
@g36 = private unnamed_addr constant [9 x i8] c"December\00", align 1
@g37 = private unnamed_addr constant [3 x i8] c"AM\00", align 1
@g38 = private unnamed_addr constant [3 x i8] c"PM\00", align 1
@g39 = private unnamed_addr constant [21 x i8] c"%a %b %e %H:%M:%S %Y\00", align 1
@g40 = private unnamed_addr constant [9 x i8] c"%m/%d/%y\00", align 1
@g41 = private unnamed_addr constant [9 x i8] c"%H:%M:%S\00", align 1
@g42 = private unnamed_addr constant [12 x i8] c"%I:%M:%S %p\00", align 1
@g43 = constant %s.0 { [7 x ptr] [ptr @g0, ptr @g1, ptr @g2, ptr @g3, ptr @g4, ptr @g5, ptr @g6], [7 x ptr] [ptr @g7, ptr @g8, ptr @g9, ptr @g10, ptr @g11, ptr @g12, ptr @g13], [12 x ptr] [ptr @g14, ptr @g15, ptr @g16, ptr @g17, ptr @g18, ptr @g19, ptr @g20, ptr @g21, ptr @g22, ptr @g23, ptr @g24, ptr @g25], [12 x ptr] [ptr @g26, ptr @g27, ptr @g28, ptr @g29, ptr @g18, ptr @g30, ptr @g31, ptr @g32, ptr @g33, ptr @g34, ptr @g35, ptr @g36], [2 x ptr] [ptr @g37, ptr @g38], ptr @g39, ptr @g40, ptr @g41, ptr @g42 }, align 4
@g44 = global ptr @g43, align 4
@g45 = private unnamed_addr constant [6 x i8] c"%H:%M\00", align 1

; Function Attrs: nounwind readonly
define ptr @f0(ptr readonly %a0, ptr nocapture readonly %a1, ptr readonly %a2) #0 {
b0:
  %v0 = icmp eq ptr %a0, null
  br i1 %v0, label %b15, label %b1

b1:                                               ; preds = %b0
  %v1 = load ptr, ptr @g44, align 4, !tbaa !0
  %v2 = getelementptr inbounds %s.0, ptr %v1, i32 0, i32 5
  %v3 = getelementptr inbounds %s.0, ptr %v1, i32 0, i32 6
  br label %b2

b2:                                               ; preds = %b14, %b6, %b1
  %v4 = phi i32 [ undef, %b1 ], [ %v31, %b14 ], [ 0, %b6 ]
  %v5 = phi ptr [ %a0, %b1 ], [ %v30, %b14 ], [ %v18, %b6 ]
  %v6 = phi ptr [ %a1, %b1 ], [ %v13, %b14 ], [ %v13, %b6 ]
  %v7 = load i8, ptr %v6, align 1, !tbaa !4
  %v8 = icmp eq i8 %v7, 0
  br i1 %v8, label %b15, label %b3

b3:                                               ; preds = %b2
  %v9 = getelementptr inbounds i8, ptr %v6, i32 1
  br label %b4

b4:                                               ; preds = %b7, %b3
  %v10 = phi ptr [ %v6, %b3 ], [ %v11, %b7 ]
  %v11 = phi ptr [ %v9, %b3 ], [ %v13, %b7 ]
  %v12 = phi i32 [ %v4, %b3 ], [ %v21, %b7 ]
  %v13 = getelementptr inbounds i8, ptr %v10, i32 2
  %v14 = load i8, ptr %v11, align 1, !tbaa !4
  %v15 = zext i8 %v14 to i32
  switch i32 %v15, label %b15 [
    i32 37, label %b5
    i32 69, label %b7
    i32 79, label %b8
    i32 99, label %b13
    i32 68, label %b9
    i32 82, label %b10
    i32 120, label %b12
  ]

b5:                                               ; preds = %b4
  %v16 = load i8, ptr %v5, align 1, !tbaa !4
  %v17 = icmp eq i8 %v14, %v16
  br i1 %v17, label %b6, label %b15

b6:                                               ; preds = %b5
  %v18 = getelementptr inbounds i8, ptr %v5, i32 1
  %v19 = icmp eq i32 %v12, 0
  br i1 %v19, label %b2, label %b15

b7:                                               ; preds = %b10, %b9, %b8, %b4
  %v20 = phi ptr [ blockaddress(@f0, %b4), %b8 ], [ blockaddress(@f0, %b11), %b9 ], [ blockaddress(@f0, %b11), %b10 ], [ blockaddress(@f0, %b4), %b4 ]
  %v21 = phi i32 [ 2, %b8 ], [ 1, %b9 ], [ 1, %b10 ], [ 1, %b4 ]
  %v22 = phi ptr [ @g40, %b8 ], [ @g40, %b9 ], [ @g45, %b10 ], [ @g40, %b4 ]
  %v23 = icmp eq i32 %v12, 0
  %v24 = select i1 %v23, ptr %v20, ptr blockaddress(@f0, %b15)
  indirectbr ptr %v24, [label %b4, label %b11, label %b15]

b8:                                               ; preds = %b4
  br label %b7

b9:                                               ; preds = %b4
  br label %b7

b10:                                              ; preds = %b4
  br label %b7

b11:                                              ; preds = %b7
  %v25 = tail call ptr @f0(ptr %v5, ptr %v22, ptr %a2) #1
  br label %b14

b12:                                              ; preds = %b4
  br label %b13

b13:                                              ; preds = %b12, %b4
  %v26 = phi ptr [ %v3, %b12 ], [ %v2, %b4 ]
  %v27 = load ptr, ptr %v26, align 4
  %v28 = tail call ptr @f0(ptr %v5, ptr %v27, ptr %a2) #1
  %v29 = icmp ugt i32 %v12, 1
  br i1 %v29, label %b15, label %b14

b14:                                              ; preds = %b13, %b11
  %v30 = phi ptr [ %v28, %b13 ], [ %v25, %b11 ]
  %v31 = phi i32 [ %v12, %b13 ], [ 0, %b11 ]
  %v32 = icmp eq ptr %v30, null
  br i1 %v32, label %b15, label %b2

b15:                                              ; preds = %b14, %b13, %b7, %b6, %b5, %b4, %b2, %b0
  %v33 = phi ptr [ null, %b0 ], [ null, %b4 ], [ null, %b7 ], [ null, %b13 ], [ null, %b14 ], [ %v5, %b2 ], [ null, %b5 ], [ null, %b6 ]
  ret ptr %v33
}

attributes #0 = { nounwind readonly }
attributes #1 = { nobuiltin nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!2, !2, i64 0}
