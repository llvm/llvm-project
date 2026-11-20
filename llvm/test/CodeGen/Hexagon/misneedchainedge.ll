; RUN: llc -march=hexagon -prevent-duplex-separation=false < %s | FileCheck %s
; CHECK: [[ADDR:r[0-9]+]] = addasl([[BASE:r[0-9]+]],[[INDEX:r[0-9]+]],#3)
; CHECK: [[REG:r[0-9]+]] = ##57005
; CHECK: memw([[BASE]]+[[INDEX]]<<#3) = [[REG]]
; CHECK: memw([[ADDR]]+#4) = #79

%struct.M = type { i32, i32 }

define i32 @foo(ptr nocapture %n, i32 %a) #0 {
entry:
  %tobool = icmp eq i32 %a, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = load ptr, ptr %n, align 4, !tbaa !0
  %x = getelementptr inbounds %struct.M, ptr %0, i32 %a, i32 0
  store i32 57005, ptr %x, align 4, !tbaa !4
  %y = getelementptr inbounds %struct.M, ptr %0, i32 %a, i32 1
  store i32 79, ptr %y, align 4, !tbaa !4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %1 = load ptr, ptr %n, align 4, !tbaa !0
  %z = getelementptr inbounds %struct.M, ptr %1, i32 %a, i32 1
  %2 = load i32, ptr %z, align 4, !tbaa !4
  ret i32 %2
}

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !2}
