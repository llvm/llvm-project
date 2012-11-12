; RUN: opt %s -basicaa -gvn -S -o - | FileCheck %s

define i32 @test1(i32* %p) {
; CHECK: @test1(i32* %p)
; CHECK: %a = load i32* %p, !range !0
; CHECK: %c = add i32 %a, %a
  %a = load i32* %p, !range !0
  %b = load i32* %p, !range !0
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test2(i32* %p) {
; CHECK: @test2(i32* %p)
; CHECK: %a = load i32* %p
; CHECK-NOT: range
; CHECK: %c = add i32 %a, %a
  %a = load i32* %p, !range !0
  %b = load i32* %p
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test3(i32* %p) {
; CHECK: @test3(i32* %p)
; CHECK: %a = load i32* %p, !range ![[DISJOINT_RANGE:[0-9]+]]
; CHECK: %c = add i32 %a, %a
  %a = load i32* %p, !range !0
  %b = load i32* %p, !range !1
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test4(i32* %p) {
; CHECK: @test4(i32* %p)
; CHECK: %a = load i32* %p, !range ![[MERGED_RANGE:[0-9]+]]
; CHECK: %c = add i32 %a, %a
  %a = load i32* %p, !range !0
  %b = load i32* %p, !range !2
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test5(i32* %p) {
; CHECK: @test5(i32* %p)
; CHECK: %a = load i32* %p, !range ![[MERGED_SIGNED_RANGE:[0-9]+]]
; CHECK: %c = add i32 %a, %a
  %a = load i32* %p, !range !3
  %b = load i32* %p, !range !4
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test6(i32* %p) {
; CHECK: @test6(i32* %p)
; CHECK: %a = load i32* %p, !range ![[MERGED_TEST6:[0-9]+]]
; CHECK: %c = add i32 %a, %a
  %a = load i32* %p, !range !5
  %b = load i32* %p, !range !6
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test7(i32* %p) {
; CHECK: @test7(i32* %p)
; CHECK: %a = load i32* %p, !range ![[MERGED_TEST7:[0-9]+]]
; CHECK: %c = add i32 %a, %a
  %a = load i32* %p, !range !7
  %b = load i32* %p, !range !8
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @test8(i32* %p) {
; CHECK: @test8(i32* %p)
; CHECK: %a = load i32* %p
; CHECK-NOT: range
; CHECK: %c = add i32 %a, %a
  %a = load i32* %p, !range !9
  %b = load i32* %p, !range !10
  %c = add i32 %a, %b
  ret i32 %c
}

; CHECK: ![[DISJOINT_RANGE]] = metadata !{i32 0, i32 2, i32 3, i32 5}
; CHECK: ![[MERGED_RANGE]] = metadata !{i32 0, i32 5}
; CHECK: ![[MERGED_SIGNED_RANGE]] = metadata !{i32 -3, i32 -2, i32 1, i32 2}
; CHECK: ![[MERGED_TEST6]] = metadata !{i32 10, i32 1}
; CHECK: ![[MERGED_TEST7]] = metadata !{i32 3, i32 4, i32 5, i32 2}

!0 = metadata !{i32 0, i32 2}
!1 = metadata !{i32 3, i32 5}
!2 = metadata !{i32 2, i32 5}
!3 = metadata !{i32 -3, i32 -2}
!4 = metadata !{i32 1, i32 2}
!5 = metadata !{i32 10, i32 1}
!6 = metadata !{i32 12, i32 13}
!7 = metadata !{i32 1, i32 2, i32 3, i32 4}
!8 = metadata !{i32 5, i32 1}
!9 = metadata !{i32 1, i32 5}
!10 = metadata !{i32 5, i32 1}
