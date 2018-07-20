; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

target triple = "hexagon"

%s.0 = type { i8, i8, i8, i8 }
%s.1 = type { %s.2 }
%s.2 = type { %s.3 }
%s.3 = type { i32 (...)** }
%s.4 = type { i8, i8, i16, i8 }
%s.5 = type { i8, %s.0* }

@g0 = external hidden global [3 x %s.0], align 8
@g1 = external hidden global [3 x %s.0], align 8
@g2 = external hidden global [3 x %s.0], align 8
@g3 = external hidden global [3 x %s.0], align 8
@g4 = external hidden global [3 x %s.0], align 8
@g5 = external hidden global [3 x %s.0], align 8
@g6 = external hidden global [4 x %s.0], align 8
@g7 = external hidden global [3 x %s.0], align 8
@g8 = external hidden global [3 x %s.0], align 8
@g9 = external hidden global [3 x %s.0], align 8
@g10 = external hidden global [4 x %s.0], align 8
@g11 = external hidden global [3 x %s.0], align 8
@g12 = external hidden global [3 x %s.0], align 8
@g13 = external hidden global [4 x %s.0], align 8
@g14 = external hidden global [3 x %s.0], align 8
@g15 = external hidden global [3 x %s.0], align 8
@g16 = external hidden global [3 x %s.0], align 8
@g17 = external hidden global [4 x %s.0], align 8
@g18 = external hidden global [3 x %s.0], align 8

; Function Attrs: norecurse nounwind optsize ssp
define hidden zeroext i8 @f0(%s.1* nocapture readnone %a0, %s.4* readonly %a1, %s.5* %a2, i32 %a3) unnamed_addr #0 align 2 {
b0:
  br i1 undef, label %b4, label %b1

b1:                                               ; preds = %b0
  %v0 = icmp eq i32 %a3, 1
  %v1 = select i1 %v0, %s.0* getelementptr inbounds ([3 x %s.0], [3 x %s.0]* @g18, i32 0, i32 0), %s.0* getelementptr inbounds ([3 x %s.0], [3 x %s.0]* @g0, i32 0, i32 0)
  %v2 = icmp eq i32 %a3, 2
  %v3 = select i1 %v2, %s.0* getelementptr inbounds ([3 x %s.0], [3 x %s.0]* @g16, i32 0, i32 0), %s.0* %v1
  %v4 = icmp eq i32 %a3, 3
  %v5 = select i1 %v4, %s.0* getelementptr inbounds ([3 x %s.0], [3 x %s.0]* @g15, i32 0, i32 0), %s.0* %v3
  %v6 = icmp eq i32 %a3, 4
  %v7 = select i1 %v6, %s.0* getelementptr inbounds ([3 x %s.0], [3 x %s.0]* @g14, i32 0, i32 0), %s.0* %v5
  %v8 = icmp eq i32 %a3, 5
  %v9 = select i1 %v8, %s.0* getelementptr inbounds ([3 x %s.0], [3 x %s.0]* @g12, i32 0, i32 0), %s.0* %v7
  %v10 = icmp eq i32 %a3, 6
  %v11 = select i1 %v10, %s.0* getelementptr inbounds ([3 x %s.0], [3 x %s.0]* @g11, i32 0, i32 0), %s.0* %v9
  %v12 = icmp eq i32 %a3, 7
  %v13 = select i1 %v12, %s.0* getelementptr inbounds ([3 x %s.0], [3 x %s.0]* @g9, i32 0, i32 0), %s.0* %v11
  %v14 = icmp eq i32 %a3, 8
  %v15 = select i1 %v14, %s.0* getelementptr inbounds ([3 x %s.0], [3 x %s.0]* @g8, i32 0, i32 0), %s.0* %v13
  %v16 = icmp eq i32 %a3, 9
  %v17 = select i1 %v16, %s.0* getelementptr inbounds ([3 x %s.0], [3 x %s.0]* @g7, i32 0, i32 0), %s.0* %v15
  %v18 = icmp eq i32 %a3, 10
  %v19 = select i1 %v18, %s.0* getelementptr inbounds ([3 x %s.0], [3 x %s.0]* @g5, i32 0, i32 0), %s.0* %v17
  %v20 = icmp eq i32 %a3, 11
  %v21 = select i1 %v20, %s.0* getelementptr inbounds ([3 x %s.0], [3 x %s.0]* @g4, i32 0, i32 0), %s.0* %v19
  %v22 = icmp eq i32 %a3, 12
  %v23 = select i1 %v22, %s.0* getelementptr inbounds ([3 x %s.0], [3 x %s.0]* @g3, i32 0, i32 0), %s.0* %v21
  %v24 = icmp eq i32 %a3, 13
  %v25 = select i1 %v24, %s.0* getelementptr inbounds ([3 x %s.0], [3 x %s.0]* @g2, i32 0, i32 0), %s.0* %v23
  %v26 = select i1 undef, %s.0* getelementptr inbounds ([3 x %s.0], [3 x %s.0]* @g1, i32 0, i32 0), %s.0* %v25
  %v27 = select i1 undef, %s.0* getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g17, i32 0, i32 0), %s.0* %v26
  %v28 = icmp eq i32 %a3, 16
  %v29 = select i1 %v28, %s.0* getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g13, i32 0, i32 0), %s.0* %v27
  %v30 = icmp eq i32 %a3, 17
  %v31 = select i1 %v30, %s.0* null, %s.0* %v29
  %v32 = select i1 undef, %s.0* getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g10, i32 0, i32 0), %s.0* %v31
  %v33 = select i1 undef, %s.0* getelementptr inbounds ([4 x %s.0], [4 x %s.0]* @g6, i32 0, i32 0), %s.0* %v32
  %v34 = add i32 %a3, -15
  %v35 = icmp ult i32 %v34, 2
  %v36 = select i1 %v35, i8 4, i8 3
  %v37 = select i1 undef, i8 0, i8 %v36
  %v38 = select i1 undef, i8 4, i8 %v37
  br i1 undef, label %b2, label %b3

b2:                                               ; preds = %b3, %b1
  %v39 = phi %s.0* [ undef, %b3 ], [ %v33, %b1 ]
  %v40 = phi i8 [ undef, %b3 ], [ %v38, %b1 ]
  %v41 = getelementptr inbounds %s.5, %s.5* %a2, i32 0, i32 1
  store %s.0* %v39, %s.0** %v41, align 4
  store i8 %v40, i8* undef, align 4
  br label %b4

b3:                                               ; preds = %b1
  br label %b2

b4:                                               ; preds = %b2, %b0
  ret i8 undef
}

attributes #0 = { norecurse nounwind optsize ssp "target-cpu"="hexagonv55" }
