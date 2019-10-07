; RUN: llc -march=hexagon -enable-pipeliner -pipeliner-max-stages=3 < %s -pipeliner-experimental-cg=true | FileCheck %s

%s.0 = type { i16, i8, i8, i16, i8, i8, i16, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i16, i8, i8, %s.1, [2 x [16 x %s.2]], i32 (i8*, i8*, i8*, i8*, i8*)*, %s.3*, %s.3*, [120 x i8], i8, i8, %s.4*, [2 x [120 x [8 x i8]]], [56 x i8], [2 x [121 x %s.5]], [2 x %s.5], %s.5*, %s.5*, i32, i32, i16, i8, i8, %s.7, %s.9, %s.11, %s.8*, %s.8* }
%s.1 = type { i8, i8, i8, i8, i8, i8, i8, i8, i32, i8, [16 x i8], i8, [4 x i8], [32 x i16], [32 x i16], [2 x i8], [4 x i8], [2 x [4 x i8]], [2 x [4 x i8]], i32, i32, i16, i8 }
%s.2 = type { [2 x i16] }
%s.3 = type { i16*, i16*, i32, i32 }
%s.4 = type { i8*, i8*, i8*, i32, i32, i32, i32 }
%s.5 = type { %s.6, [2 x [4 x %s.2]], [2 x [2 x i8]], [2 x i8] }
%s.6 = type { i8, i8, i8, i8, i8, i8, i8, i8, i32 }
%s.7 = type { [12 x %s.8], [4 x %s.8], [2 x %s.8], [4 x %s.8], [6 x %s.8], [2 x [7 x %s.8]], [4 x %s.8], [3 x [4 x %s.8]], [3 x %s.8], [3 x %s.8] }
%s.8 = type { i8, i8 }
%s.9 = type { [371 x %s.8], [6 x %s.10] }
%s.10 = type { %s.8*, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
%s.11 = type { i32, i32, i8* }

; Function Attrs: nounwind
define void @f0(%s.0* %a0, i8* %a1, i16* %a2, i16** %a3, i16** %a4, i32 %a5) #0 {
b0:
  %v0 = load i8, i8* %a1, align 1, !tbaa !0
  %v1 = icmp eq i8 %v0, 1
  br i1 %v1, label %b1, label %b2

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: memh([[REG0:(r[0-9]+)]]+#0) = #0
; CHECK: }{{[ \t]*}}:endloop0

b1:                                               ; preds = %b1, %b0
  %v2 = phi i16* [ %v17, %b1 ], [ %a2, %b0 ]
  %v3 = phi i32 [ %v18, %b1 ], [ 0, %b0 ]
  %v4 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 25, i32 10, i32 %v3
  %v5 = load i8, i8* %v4, align 1, !tbaa !0
  %v6 = zext i8 %v5 to i16
  %v7 = add nsw i32 %v3, 1
  %v8 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 25, i32 10, i32 %v7
  %v9 = load i8, i8* %v8, align 1, !tbaa !0
  %v10 = or i16 0, %v6
  %v11 = load i8, i8* %a1, align 1, !tbaa !0
  %v12 = zext i8 %v11 to i16
  %v13 = shl nuw i16 %v12, 8
  %v14 = or i16 %v10, %v13
  %v15 = or i16 %v14, 0
  %v16 = getelementptr inbounds i16, i16* %v2, i32 1
  store i16* %v16, i16** %a3, align 4, !tbaa !3
  store i16 %v15, i16* %v2, align 2, !tbaa !5
  %v17 = getelementptr inbounds i16, i16* %v2, i32 2
  store i16* %v17, i16** %a4, align 4, !tbaa !3
  store i16 0, i16* %v16, align 2, !tbaa !5
  %v18 = add nsw i32 %v3, 8
  %v19 = icmp slt i32 %v18, %a5
  br i1 %v19, label %b1, label %b2

b2:                                               ; preds = %b1, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!4, !4, i64 0}
!4 = !{!"any pointer", !1}
!5 = !{!6, !6, i64 0}
!6 = !{!"short", !1}
