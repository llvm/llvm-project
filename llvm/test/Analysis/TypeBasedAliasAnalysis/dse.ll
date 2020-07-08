; RUN: opt < %s -tbaa -basic-aa -dse -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; DSE should make use of TBAA.

; CHECK: @test0_yes
; CHECK-NEXT: load i8, i8* %b
; CHECK-NEXT: store i8 1, i8* %a
; CHECK-NEXT: ret i8 %y
define i8 @test0_yes(i8* %a, i8* %b) nounwind {
  store i8 0, i8* %a, !tbaa !1
  %y = load i8, i8* %b, !tbaa !2
  store i8 1, i8* %a, !tbaa !1
  ret i8 %y
}

; CHECK: @test0_no
; CHECK-NEXT: store i8 0, i8* %a
; CHECK-NEXT: load i8, i8* %b
; CHECK-NEXT: store i8 1, i8* %a
; CHECK-NEXT: ret i8 %y
define i8 @test0_no(i8* %a, i8* %b) nounwind {
  store i8 0, i8* %a, !tbaa !3
  %y = load i8, i8* %b, !tbaa !4
  store i8 1, i8* %a, !tbaa !3
  ret i8 %y
}

; CHECK: @test1_yes
; CHECK-NEXT: load i8, i8* %b
; CHECK-NEXT: store i8 1, i8* %a
; CHECK-NEXT: ret i8 %y
define i8 @test1_yes(i8* %a, i8* %b) nounwind {
  store i8 0, i8* %a
  %y = load i8, i8* %b, !tbaa !5
  store i8 1, i8* %a
  ret i8 %y
}

; CHECK: @test1_no
; CHECK-NEXT: store i8 0, i8* %a
; CHECK-NEXT: load i8, i8* %b
; CHECK-NEXT: store i8 1, i8* %a
; CHECK-NEXT: ret i8 %y
define i8 @test1_no(i8* %a, i8* %b) nounwind {
  store i8 0, i8* %a
  %y = load i8, i8* %b, !tbaa !6
  store i8 1, i8* %a
  ret i8 %y
}

; Root note.
!0 = !{ }
; Some type.
!1 = !{!7, !7, i64 0}
; Some other non-aliasing type.
!2 = !{!8, !8, i64 0}

; Some type.
!3 = !{!9, !9, i64 0}
; Some type in a different type system.
!4 = !{!10, !10, i64 0}

; Invariant memory.
!5 = !{!11, !11, i64 0, i1 1}
; Not invariant memory.
!6 = !{!11, !11, i64 0, i1 0}
!7 = !{ !"foo", !0 }
!8 = !{ !"bar", !0 }
!9 = !{ !"foo", !0 }
!10 = !{ !"bar", !12}
!11 = !{ !"qux", !0}
!12 = !{!"different"}
