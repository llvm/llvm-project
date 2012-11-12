; RUN: opt < %s -instcombine -S | FileCheck %s

; Test an already canonical branch to make sure we don't flip those.
define i32 @test0(i32 %X, i32 %Y) {
        %C = icmp eq i32 %X, %Y
        br i1 %C, label %T, label %F, !prof !0

; CHECK: @test0
; CHECK: %C = icmp eq i32 %X, %Y
; CHECK: br i1 %C, label %T, label %F

T:
        ret i32 12
F:
        ret i32 123
}

define i32 @test1(i32 %X, i32 %Y) {
        %C = icmp ne i32 %X, %Y
        br i1 %C, label %T, label %F, !prof !1

; CHECK: @test1
; CHECK: %C = icmp eq i32 %X, %Y
; CHECK: br i1 %C, label %F, label %T

T:
        ret i32 12
F:
        ret i32 123
}

define i32 @test2(i32 %X, i32 %Y) {
        %C = icmp ule i32 %X, %Y
        br i1 %C, label %T, label %F, !prof !2

; CHECK: @test2
; CHECK: %C = icmp ugt i32 %X, %Y
; CHECK: br i1 %C, label %F, label %T

T:
        ret i32 12
F:
        ret i32 123
}

define i32 @test3(i32 %X, i32 %Y) {
        %C = icmp uge i32 %X, %Y
        br i1 %C, label %T, label %F, !prof !3

; CHECK: @test3
; CHECK: %C = icmp ult i32 %X, %Y
; CHECK: br i1 %C, label %F, label %T

T:
        ret i32 12
F:
        ret i32 123
}

!0 = metadata !{metadata !"branch_weights", i32 1, i32 2}
!1 = metadata !{metadata !"branch_weights", i32 3, i32 4}
!2 = metadata !{metadata !"branch_weights", i32 5, i32 6}
!3 = metadata !{metadata !"branch_weights", i32 7, i32 8}
; Base case shouldn't change.
; CHECK: !0 = {{.*}} i32 1, i32 2}
; Ensure that the branch metadata is reversed to match the reversals above.
; CHECK: !1 = {{.*}} i32 4, i32 3}
; CHECK: !2 = {{.*}} i32 6, i32 5}
; CHECK: !3 = {{.*}} i32 8, i32 7}
