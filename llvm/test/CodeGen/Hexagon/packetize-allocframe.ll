; RUN: llc -mtriple=hexagon -O2 -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

; The purpose of this test is to make sure that the packetizer is ignoring
; CFI instructions while forming packet for allocframe. Refer to 7d7d99622
; which replaced PROLOG_LABEL with CFI_INSTRUCTION.

@g0 = external constant ptr

; We used to emit:
;      {
;        allocframe(#0)
;      }
;      {
;         r0 = #4
; But we can put more instructions in the first packet.

; CHECK:      {
; CHECK-NEXT:   call f1
; CHECK-NEXT:   r0 = #4
; CHECK-NEXT:   allocframe(#0)
; CHECK-NEXT: }

define i32 @f0() personality ptr @f3 {
b0:
  %v0 = tail call ptr @f1(i32 4) #1
  store i32 20, ptr %v0, align 4, !tbaa !0
  invoke void @f2(ptr %v0, ptr @g0, ptr null) #2
          to label %b4 unwind label %b1

b1:                                               ; preds = %b0
  %v2 = landingpad { ptr, i32 }
          catch ptr @g0
  %v3 = extractvalue { ptr, i32 } %v2, 1
  %v4 = tail call i32 @llvm.eh.typeid.for(ptr @g0) #1
  %v5 = icmp eq i32 %v3, %v4
  br i1 %v5, label %b2, label %b3

b2:                                               ; preds = %b1
  %v6 = extractvalue { ptr, i32 } %v2, 0
  %v7 = tail call ptr @f4(ptr %v6) #1
  tail call void @f5() #1
  ret i32 1

b3:                                               ; preds = %b1
  resume { ptr, i32 } %v2

b4:                                               ; preds = %b0
  unreachable
}

declare ptr @f1(i32)

declare void @f2(ptr, ptr, ptr)

declare i32 @f3(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(ptr) #0

declare ptr @f4(ptr)

declare void @f5()

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { noreturn }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
