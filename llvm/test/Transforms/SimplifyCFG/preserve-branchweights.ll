; RUN: opt -simplifycfg -S -o - < %s | FileCheck %s

declare void @helper(i32)

define void @test1(i1 %a, i1 %b) {
; CHECK: @test1
entry:
  br i1 %a, label %Y, label %X, !prof !0
; CHECK: br i1 %or.cond, label %Z, label %Y, !prof !0

X:
  %c = or i1 %b, false
  br i1 %c, label %Z, label %Y, !prof !1

Y:
  call void @helper(i32 0)
  ret void

Z:
  call void @helper(i32 1)
  ret void
}

define void @test2(i1 %a, i1 %b) {
; CHECK: @test2
entry:
  br i1 %a, label %X, label %Y, !prof !1
; CHECK: br i1 %or.cond, label %Z, label %Y, !prof !1
; CHECK-NOT: !prof

X:
  %c = or i1 %b, false
  br i1 %c, label %Z, label %Y, !prof !2

Y:
  call void @helper(i32 0)
  ret void

Z:
  call void @helper(i32 1)
  ret void
}

define void @test3(i1 %a, i1 %b) {
; CHECK: @test3
; CHECK-NOT: !prof
entry:
  br i1 %a, label %X, label %Y, !prof !1

X:
  %c = or i1 %b, false
  br i1 %c, label %Z, label %Y

Y:
  call void @helper(i32 0)
  ret void

Z:
  call void @helper(i32 1)
  ret void
}

define void @test4(i1 %a, i1 %b) {
; CHECK: @test4
; CHECK-NOT: !prof
entry:
  br i1 %a, label %X, label %Y

X:
  %c = or i1 %b, false
  br i1 %c, label %Z, label %Y, !prof !1

Y:
  call void @helper(i32 0)
  ret void

Z:
  call void @helper(i32 1)
  ret void
}

;; test5 - The case where it jumps to the default target will be removed.
define void @test5(i32 %M, i32 %N) nounwind uwtable {
entry:
  switch i32 %N, label %sw2 [
    i32 1, label %sw2
    i32 2, label %sw.bb
    i32 3, label %sw.bb1
  ], !prof !3
; CHECK: test5
; CHECK: switch i32 %N, label %sw2 [
; CHECK: i32 3, label %sw.bb1
; CHECK: i32 2, label %sw.bb
; CHECK: ], !prof !2

sw.bb:
  call void @helper(i32 0)
  br label %sw.epilog

sw.bb1:
  call void @helper(i32 1)
  br label %sw.epilog

sw2:
  call void @helper(i32 2)
  br label %sw.epilog

sw.epilog:
  ret void
}

;; test6 - Some cases of the second switch are pruned during optimization.
;; Then the second switch will be converted to a branch, finally, the first
;; switch and the branch will be merged into a single switch.
define void @test6(i32 %M, i32 %N) nounwind uwtable {
entry:
  switch i32 %N, label %sw2 [
    i32 1, label %sw2
    i32 2, label %sw.bb
    i32 3, label %sw.bb1
  ], !prof !4
; CHECK: test6
; CHECK: switch i32 %N, label %sw.epilog
; CHECK: i32 3, label %sw.bb1
; CHECK: i32 2, label %sw.bb
; CHECK: i32 4, label %sw.bb5
; CHECK: ], !prof !3

sw.bb:
  call void @helper(i32 0)
  br label %sw.epilog

sw.bb1:
  call void @helper(i32 1)
  br label %sw.epilog

sw2:
;; Here "case 2" is invalidated since the default case of the first switch
;; does not include "case 2".
  switch i32 %N, label %sw.epilog [
    i32 2, label %sw.bb4
    i32 4, label %sw.bb5
  ], !prof !5

sw.bb4:
  call void @helper(i32 2)
  br label %sw.epilog

sw.bb5:
  call void @helper(i32 3)
  br label %sw.epilog

sw.epilog:
  ret void
}

;; This test is based on test1 but swapped the targets of the second branch.
define void @test1_swap(i1 %a, i1 %b) {
; CHECK: @test1_swap
entry:
  br i1 %a, label %Y, label %X, !prof !0
; CHECK: br i1 %or.cond, label %Y, label %Z, !prof !4

X:
  %c = or i1 %b, false
  br i1 %c, label %Y, label %Z, !prof !1

Y:
  call void @helper(i32 0)
  ret void

Z:
  call void @helper(i32 1)
  ret void
}

define void @test7(i1 %a, i1 %b) {
; CHECK: @test7
entry:
  %c = or i1 %b, false
  br i1 %a, label %Y, label %X, !prof !0
; CHECK: br i1 %brmerge, label %Y, label %Z, !prof !5

X:
  br i1 %c, label %Y, label %Z, !prof !6

Y:
  call void @helper(i32 0)
  ret void

Z:
  call void @helper(i32 1)
  ret void
}

; Test basic folding to a conditional branch.
define void @test8(i64 %x, i64 %y) nounwind {
; CHECK: @test8
entry:
    %lt = icmp slt i64 %x, %y
; CHECK: br i1 %lt, label %a, label %b, !prof !6
    %qux = select i1 %lt, i32 0, i32 2
    switch i32 %qux, label %bees [
        i32 0, label %a
        i32 1, label %b
        i32 2, label %b
    ], !prof !7
a:
    call void @helper(i32 0) nounwind
    ret void
b:
    call void @helper(i32 1) nounwind
    ret void
bees:
    call void @helper(i32 2) nounwind
    ret void
}

; Test edge splitting when the default target has icmp and unconditinal
; branch
define i1 @test9(i32 %x, i32 %y) nounwind {
; CHECK: @test9
entry:
    switch i32 %x, label %bees [
        i32 0, label %a
        i32 1, label %end
        i32 2, label %end
    ], !prof !7
; CHECK: switch i32 %x, label %bees [
; CHECK: i32 0, label %a
; CHECK: i32 1, label %end
; CHECK: i32 2, label %end
; CHECK: i32 92, label %end
; CHECK: ], !prof !7

a:
    call void @helper(i32 0) nounwind
    %reta = icmp slt i32 %x, %y
    ret i1 %reta

bees:
    %tmp = icmp eq i32 %x, 92
    br label %end

end:
; CHECK: end:
; CHECK: %ret = phi i1 [ true, %entry ], [ false, %bees ], [ true, %entry ], [ true, %entry ]
    %ret = phi i1 [ true, %entry ], [%tmp, %bees], [true, %entry]
    call void @helper(i32 2) nounwind
    ret i1 %ret
}

define void @test10(i32 %x) nounwind readnone ssp noredzone {
entry:
 switch i32 %x, label %lor.rhs [
   i32 2, label %lor.end
   i32 1, label %lor.end
   i32 3, label %lor.end
 ], !prof !7

lor.rhs:
 call void @helper(i32 1) nounwind
 ret void

lor.end:
 call void @helper(i32 0) nounwind
 ret void

; CHECK: test10
; CHECK: %x.off = add i32 %x, -1
; CHECK: %switch = icmp ult i32 %x.off, 3
; CHECK: br i1 %switch, label %lor.end, label %lor.rhs, !prof !8
}

; Remove dead cases from the switch.
define void @test11(i32 %x) nounwind {
  %i = shl i32 %x, 1
  switch i32 %i, label %a [
    i32 21, label %b
    i32 24, label %c
  ], !prof !8
; CHECK: %cond = icmp eq i32 %i, 24
; CHECK: br i1 %cond, label %c, label %a, !prof !9

a:
 call void @helper(i32 0) nounwind
 ret void
b:
 call void @helper(i32 1) nounwind
 ret void
c:
 call void @helper(i32 2) nounwind
 ret void
}

!0 = metadata !{metadata !"branch_weights", i32 3, i32 5}
!1 = metadata !{metadata !"branch_weights", i32 1, i32 1}
!2 = metadata !{metadata !"branch_weights", i32 1, i32 2}
!3 = metadata !{metadata !"branch_weights", i32 4, i32 3, i32 2, i32 1}
!4 = metadata !{metadata !"branch_weights", i32 4, i32 3, i32 2, i32 1}
!5 = metadata !{metadata !"branch_weights", i32 7, i32 6, i32 5}
!6 = metadata !{metadata !"branch_weights", i32 1, i32 3}
!7 = metadata !{metadata !"branch_weights", i32 33, i32 9, i32 8, i32 7}
!8 = metadata !{metadata !"branch_weights", i32 33, i32 9, i32 8}

; CHECK: !0 = metadata !{metadata !"branch_weights", i32 5, i32 11}
; CHECK: !1 = metadata !{metadata !"branch_weights", i32 1, i32 5}
; CHECK: !2 = metadata !{metadata !"branch_weights", i32 7, i32 1, i32 2}
; CHECK: !3 = metadata !{metadata !"branch_weights", i32 49, i32 12, i32 24, i32 35}
; CHECK: !4 = metadata !{metadata !"branch_weights", i32 11, i32 5}
; CHECK: !5 = metadata !{metadata !"branch_weights", i32 17, i32 15} 
; CHECK: !6 = metadata !{metadata !"branch_weights", i32 9, i32 7}
; CHECK: !7 = metadata !{metadata !"branch_weights", i32 17, i32 9, i32 8, i32 7, i32 17}
; CHECK: !8 = metadata !{metadata !"branch_weights", i32 24, i32 33}
; CHECK: !9 = metadata !{metadata !"branch_weights", i32 8, i32 33}
; CHECK-NOT: !9
