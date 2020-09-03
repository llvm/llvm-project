; RUN: opt < %s -tbaa -basic-aa -gvn -S | FileCheck %s

; Test that basic alias queries work.

; CHECK: @test0_yes
; CHECK: add i8 %x, %x
define i8 @test0_yes(i8* %a, i8* %b) nounwind {
  %x = load i8, i8* %a, !tbaa !1
  store i8 0, i8* %b, !tbaa !2
  %y = load i8, i8* %a, !tbaa !1
  %z = add i8 %x, %y
  ret i8 %z
}

; CHECK: @test0_no
; CHECK: add i8 %x, %y
define i8 @test0_no(i8* %a, i8* %b) nounwind {
  %x = load i8, i8* %a, !tbaa !3
  store i8 0, i8* %b, !tbaa !4
  %y = load i8, i8* %a, !tbaa !3
  %z = add i8 %x, %y
  ret i8 %z
}

; Test that basic invariant-memory queries work.

; CHECK: @test1_yes
; CHECK: add i8 %x, %x
define i8 @test1_yes(i8* %a, i8* %b) nounwind {
  %x = load i8, i8* %a, !tbaa !5
  store i8 0, i8* %b
  %y = load i8, i8* %a, !tbaa !5
  %z = add i8 %x, %y
  ret i8 %z
}

; CHECK: @test1_no
; CHECK: add i8 %x, %y
define i8 @test1_no(i8* %a, i8* %b) nounwind {
  %x = load i8, i8* %a, !tbaa !6
  store i8 0, i8* %b
  %y = load i8, i8* %a, !tbaa !6
  %z = add i8 %x, %y
  ret i8 %z
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
!10 = !{ !"bar", !12 }
!11 = !{ !"qux", !0}
!12 = !{!"different"}
