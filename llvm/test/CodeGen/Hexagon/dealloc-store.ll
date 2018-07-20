; RUN: llc -march=hexagon -O2 -hexagon-shrink-frame=0 -hexagon-cext-threshold=1 < %s | FileCheck %s

target triple = "hexagon"

%s.0 = type <{ i8*, i8*, i16, i8, i8, i8 }>
%s.1 = type { %s.2, [14 x %s.6*], [14 x i8], [6 x i8], [4 x %s.4], [4 x %s.8], [4 x %s.8], [14 x %s.10], %s.6*, %s.6* }
%s.2 = type { [4 x %s.3], i16, i32, i32, i32, i32 }
%s.3 = type { i8, i8, i8, i8 }
%s.4 = type { i8, i32, [16 x %s.5], %s.6, i8, [7 x i8] }
%s.5 = type { void (i8*)*, i8*, i32 }
%s.6 = type { %s.7*, i32, %s.7*, i32, i32, i32, %s.7*, %s.7*, i32, i8, i32*, i32, i32, i32*, i32*, i32, i8, i32*, i32, %s.5*, i32, i32, i32, void (%s.6*)*, i32, i8 }
%s.7 = type { i32, i16, i16 }
%s.8 = type { %s.9 }
%s.9 = type { i8*, i32, i32 }
%s.10 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }

@g0 = internal constant %s.0 <{ i8* getelementptr inbounds ([125 x i8], [125 x i8]* @g1, i32 0, i32 0), i8* getelementptr inbounds ([82 x i8], [82 x i8]* @g2, i32 0, i32 0), i16 1694, i8 4, i8 0, i8 0 }>, section ".rodata.diag", align 1
@g1 = private unnamed_addr constant [125 x i8] c"............................................................................................................................\00", align 8
@g2 = private unnamed_addr constant [82 x i8] c"Assertion (..............................................................) failed\00", align 8
@g3 = external global %s.1

define void @f0(%s.6* %a0, i8 zeroext %a1) {
;  look for a dealloc_return in a packet with nothing else.
;
; CHECK: if (p{{[0-3]}}) memw(
; CHECK: }
; CHECK: {
; CHECK-NEXT: dealloc_return
; CHECK-NEXT: }
b0:
  %v0 = add i8 %a1, -2
  %v1 = icmp ugt i8 %v0, 1
  br i1 %v1, label %b1, label %b2, !prof !0

b1:                                               ; preds = %b0
  tail call void @f1(%s.0* @g0) #1
  unreachable

b2:                                               ; preds = %b0
  %v2 = icmp eq i8 %a1, 2
  br i1 %v2, label %b3, label %b4

b3:                                               ; preds = %b2
  store %s.6* %a0, %s.6** getelementptr inbounds (%s.1, %s.1* @g3, i32 0, i32 8), align 4, !tbaa !1
  br label %b5

b4:                                               ; preds = %b2
  store %s.6* %a0, %s.6** getelementptr inbounds (%s.1, %s.1* @g3, i32 0, i32 9), align 4, !tbaa !1
  br label %b5

b5:                                               ; preds = %b4, %b3
  ret void
}

; Function Attrs: noreturn
declare void @f1(%s.0*) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { noreturn }

!0 = !{!"branch_weights", i32 4, i32 64}
!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3}
!3 = !{!"omnipotent char", !4}
!4 = !{!"Simple C/C++ TBAA"}
