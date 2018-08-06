; RUN: llc -march=hexagon -O0 < %s | FileCheck %s

; We do not want to see a 'cannot select' error,
; we would like to see a vasrh instruction
; CHECK: vasrh

target triple = "hexagon"

@g0 = global [6 x i64] [i64 0, i64 1, i64 10000, i64 -9223372036854775808, i64 9223372036854775807, i64 -1], align 8
@g1 = common global i32 0, align 4

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = load i64, i64* getelementptr inbounds ([6 x i64], [6 x i64]* @g0, i32 0, i32 0), align 8, !tbaa !0
  %v1 = tail call i64 @llvm.hexagon.S2.asr.i.vh(i64 %v0, i32 62)
  %v2 = trunc i64 %v1 to i32
  store i32 %v2, i32* @g1, align 4, !tbaa !4
  ret i32 0
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.S2.asr.i.vh(i64, i32) #1

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
attributes #1 = { nounwind readnone }

!0 = !{!1, !1, i64 0}
!1 = !{!"long long", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !2, i64 0}
