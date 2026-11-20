; If conversion would turn the branch into a conditional return, leaving
; nothing to pull the load into.
; RUN: llc -march=hexagon -prevent-duplex-sep-v2=false -ifcvt-limit=0 -mno-pairing -mno-compound < %s | FileCheck %s
; RUN: llc -march=hexagon -mno-pairing -mno-compound < %s | FileCheck --check-prefix=CHECK1 %s
;; This test checks whether duplex instructions are preserved.
; CHECK: p{{[0-9]}} = cmp.eq(r{{[0-9]*}},#0)
; CHECK: if (p{{[0-2]}}.new) jump{{.*}}
; CHECK: if (!p{{[0-2]}}.new) r{{[0-9]*}} = memw
; CHECK-NEXT: }

; When duplexes are preserved
; CHECK1: p{{[0-9]}} = cmp.eq(r{{[0-9]*}},#0)
; CHECK1: if (p{{[0-2]}}.new) jumpr{{.*}}
; CHECK1-NOT: if (!p{{[0-2]}}.new) r{{[0-9]*}} = memw
; CHECK1-LABEL: .LBB0_1
; CHECK1: r{{[0-9]*}} = memw
; CHECK1-NEXT: }

%struct.M = type { i32, i32 }

define void @foo(ptr nocapture %n, i32 %a) #0 {
entry:
  %tobool = icmp eq i32 %a, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = load ptr, ptr %n, align 4, !tbaa !0
  %x = getelementptr inbounds %struct.M, ptr %0, i32 %a, i32 0
  store i32 57005, ptr %x, align 4, !tbaa !4
  %y = getelementptr inbounds %struct.M, ptr %0, i32 %a, i32 1
  store i32 48879, ptr %y, align 4, !tbaa !4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !2}
