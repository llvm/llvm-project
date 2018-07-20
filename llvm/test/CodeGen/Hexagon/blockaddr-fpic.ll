; RUN: llc -march=hexagon -relocation-model=pic -O2 < %s | FileCheck %s
; CHECK: r{{[0-9]+}} = add(pc,##.Ltmp0@PCREL)
; CHECK-NOT: r{{[0-9]+}} = ##.Ltmp0

target triple = "hexagon"

%s.0 = type { [7 x i8*], [7 x i8*], [12 x i8*], [12 x i8*], [2 x i8*], i8*, i8*, i8*, i8* }
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
@g43 = constant %s.0 { [7 x i8*] [i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g0, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g1, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g2, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g3, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g4, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g5, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g6, i32 0, i32 0)], [7 x i8*] [i8* getelementptr inbounds ([7 x i8], [7 x i8]* @g7, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @g8, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @g9, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @g10, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @g11, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @g12, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @g13, i32 0, i32 0)], [12 x i8*] [i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g14, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g15, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g16, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g17, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g18, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g19, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g20, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g21, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g22, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g23, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g24, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g25, i32 0, i32 0)], [12 x i8*] [i8* getelementptr inbounds ([8 x i8], [8 x i8]* @g26, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @g27, i32 0, i32 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @g28, i32 0, i32 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @g29, i32 0, i32 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @g18, i32 0, i32 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @g30, i32 0, i32 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @g31, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @g32, i32 0, i32 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @g33, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @g34, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @g35, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @g36, i32 0, i32 0)], [2 x i8*] [i8* getelementptr inbounds ([3 x i8], [3 x i8]* @g37, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @g38, i32 0, i32 0)], i8* getelementptr inbounds ([21 x i8], [21 x i8]* @g39, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @g40, i32 0, i32 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @g41, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8], [12 x i8]* @g42, i32 0, i32 0) }, align 4
@g44 = global %s.0* @g43, align 4
@g45 = private unnamed_addr constant [6 x i8] c"%H:%M\00", align 1

; Function Attrs: nounwind readonly
define i8* @f0(i8* readonly %a0, i8* nocapture readonly %a1, %s.1* readonly %a2) #0 {
b0:
  %v0 = icmp eq i8* %a0, null
  br i1 %v0, label %b15, label %b1

b1:                                               ; preds = %b0
  %v1 = load %s.0*, %s.0** @g44, align 4, !tbaa !0
  %v2 = getelementptr inbounds %s.0, %s.0* %v1, i32 0, i32 5
  %v3 = getelementptr inbounds %s.0, %s.0* %v1, i32 0, i32 6
  br label %b2

b2:                                               ; preds = %b14, %b6, %b1
  %v4 = phi i32 [ undef, %b1 ], [ %v31, %b14 ], [ 0, %b6 ]
  %v5 = phi i8* [ %a0, %b1 ], [ %v30, %b14 ], [ %v18, %b6 ]
  %v6 = phi i8* [ %a1, %b1 ], [ %v13, %b14 ], [ %v13, %b6 ]
  %v7 = load i8, i8* %v6, align 1, !tbaa !4
  %v8 = icmp eq i8 %v7, 0
  br i1 %v8, label %b15, label %b3

b3:                                               ; preds = %b2
  %v9 = getelementptr inbounds i8, i8* %v6, i32 1
  br label %b4

b4:                                               ; preds = %b7, %b3
  %v10 = phi i8* [ %v6, %b3 ], [ %v11, %b7 ]
  %v11 = phi i8* [ %v9, %b3 ], [ %v13, %b7 ]
  %v12 = phi i32 [ %v4, %b3 ], [ %v21, %b7 ]
  %v13 = getelementptr inbounds i8, i8* %v10, i32 2
  %v14 = load i8, i8* %v11, align 1, !tbaa !4
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
  %v16 = load i8, i8* %v5, align 1, !tbaa !4
  %v17 = icmp eq i8 %v14, %v16
  br i1 %v17, label %b6, label %b15

b6:                                               ; preds = %b5
  %v18 = getelementptr inbounds i8, i8* %v5, i32 1
  %v19 = icmp eq i32 %v12, 0
  br i1 %v19, label %b2, label %b15

b7:                                               ; preds = %b10, %b9, %b8, %b4
  %v20 = phi i8* [ blockaddress(@f0, %b4), %b8 ], [ blockaddress(@f0, %b11), %b9 ], [ blockaddress(@f0, %b11), %b10 ], [ blockaddress(@f0, %b4), %b4 ]
  %v21 = phi i32 [ 2, %b8 ], [ 1, %b9 ], [ 1, %b10 ], [ 1, %b4 ]
  %v22 = phi i8* [ getelementptr inbounds ([9 x i8], [9 x i8]* @g40, i32 0, i32 0), %b8 ], [ getelementptr inbounds ([9 x i8], [9 x i8]* @g40, i32 0, i32 0), %b9 ], [ getelementptr inbounds ([6 x i8], [6 x i8]* @g45, i32 0, i32 0), %b10 ], [ getelementptr inbounds ([9 x i8], [9 x i8]* @g40, i32 0, i32 0), %b4 ]
  %v23 = icmp eq i32 %v12, 0
  %v24 = select i1 %v23, i8* %v20, i8* blockaddress(@f0, %b15)
  indirectbr i8* %v24, [label %b4, label %b11, label %b15]

b8:                                               ; preds = %b4
  br label %b7

b9:                                               ; preds = %b4
  br label %b7

b10:                                              ; preds = %b4
  br label %b7

b11:                                              ; preds = %b7
  %v25 = tail call i8* @f0(i8* %v5, i8* %v22, %s.1* %a2) #1
  br label %b14

b12:                                              ; preds = %b4
  br label %b13

b13:                                              ; preds = %b12, %b4
  %v26 = phi i8** [ %v3, %b12 ], [ %v2, %b4 ]
  %v27 = load i8*, i8** %v26, align 4
  %v28 = tail call i8* @f0(i8* %v5, i8* %v27, %s.1* %a2) #1
  %v29 = icmp ugt i32 %v12, 1
  br i1 %v29, label %b15, label %b14

b14:                                              ; preds = %b13, %b11
  %v30 = phi i8* [ %v28, %b13 ], [ %v25, %b11 ]
  %v31 = phi i32 [ %v12, %b13 ], [ 0, %b11 ]
  %v32 = icmp eq i8* %v30, null
  br i1 %v32, label %b15, label %b2

b15:                                              ; preds = %b14, %b13, %b7, %b6, %b5, %b4, %b2, %b0
  %v33 = phi i8* [ null, %b0 ], [ null, %b4 ], [ null, %b7 ], [ null, %b13 ], [ null, %b14 ], [ %v5, %b2 ], [ null, %b5 ], [ null, %b6 ]
  ret i8* %v33
}

attributes #0 = { nounwind readonly }
attributes #1 = { nobuiltin nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!2, !2, i64 0}
