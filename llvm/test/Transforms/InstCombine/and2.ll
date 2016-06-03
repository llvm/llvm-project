; RUN: opt < %s -instcombine -S | FileCheck %s

; PR1738
define i1 @test1(double %X, double %Y) {
        %tmp9 = fcmp ord double %X, 0.000000e+00
        %tmp13 = fcmp ord double %Y, 0.000000e+00
        %bothcond = and i1 %tmp13, %tmp9
        ret i1 %bothcond
; CHECK:  fcmp ord double %Y, %X
}

define i1 @test2(i1 %X, i1 %Y) {
  %a = and i1 %X, %Y
  %b = and i1 %a, %X
  ret i1 %b
; CHECK-LABEL: @test2(
; CHECK-NEXT: and i1 %X, %Y
; CHECK-NEXT: ret
}

define i32 @test3(i32 %X, i32 %Y) {
  %a = and i32 %X, %Y
  %b = and i32 %Y, %a
  ret i32 %b
; CHECK-LABEL: @test3(
; CHECK-NEXT: and i32 %X, %Y
; CHECK-NEXT: ret
}

define i1 @test4(i32 %X) {
  %a = icmp ult i32 %X, 31
  %b = icmp slt i32 %X, 0
  %c = and i1 %a, %b
  ret i1 %c
; CHECK-LABEL: @test4(
; CHECK-NEXT: ret i1 false
}

; Make sure we don't go into an infinite loop with this test
define <4 x i32> @test5(<4 x i32> %A) {
  %1 = xor <4 x i32> %A, <i32 1, i32 2, i32 3, i32 4>
  %2 = and <4 x i32> <i32 1, i32 2, i32 3, i32 4>, %1
  ret <4 x i32> %2
}

; Check that we combine "if x!=0 && x!=-1" into "if x+1u>1"
define i32 @test6(i64 %x) nounwind {
; CHECK-LABEL: @test6(
; CHECK-NEXT: add i64 %x, 1
; CHECK-NEXT: icmp ugt i64 %x.off, 1
  %cmp1 = icmp ne i64 %x, -1
  %not.cmp = icmp ne i64 %x, 0
  %.cmp1 = and i1 %cmp1, %not.cmp
  %land.ext = zext i1 %.cmp1 to i32
  ret i32 %land.ext
}

define i1 @test7(i32 %i, i1 %b) {
; CHECK-LABEL: @test7(
; CHECK-NEXT: [[CMP:%.*]] = icmp eq i32 %i, 0
; CHECK-NEXT: [[AND:%.*]] = and i1 [[CMP]], %b
; CHECK-NEXT: ret i1 [[AND]]
  %cmp1 = icmp slt i32 %i, 1
  %cmp2 = icmp sgt i32 %i, -1
  %and1 = and i1 %cmp1, %b
  %and2 = and i1 %and1, %cmp2
  ret i1 %and2
}

define i1 @test8(i32 %i) {
; CHECK-LABEL: @test8(
; CHECK-NEXT: [[DEC:%.*]] = add i32 %i, -1
; CHECK-NEXT: [[CMP:%.*]] = icmp ult i32 [[DEC]], 13
; CHECK-NEXT: ret i1 [[CMP]]
  %cmp1 = icmp ne i32 %i, 0
  %cmp2 = icmp ult i32 %i, 14
  %cond = and i1 %cmp1, %cmp2
  ret i1 %cond
}

; combine -x & 1 into x & 1
define i64 @test9(i64 %x) {
; CHECK-LABEL: @test9(
; CHECK-NOT: %sub = sub nsw i64 0, %x
; CHECK-NOT: %and = and i64 %sub, 1
; CHECK-NEXT: %and = and i64 %x, 1
; CHECK-NEXT: ret i64 %and
  %sub = sub nsw i64 0, %x
  %and = and i64 %sub, 1
  ret i64 %and
}

define i64 @test10(i64 %x) {
; CHECK-LABEL: @test10(
; CHECK-NOT: %sub = sub nsw i64 0, %x
; CHECK-NEXT: %and = and i64 %x, 1
; CHECK-NOT: %add = add i64 %sub, %and
; CHECK-NEXT: %add = sub i64 %and, %x
; CHECK-NEXT: ret i64 %add
  %sub = sub nsw i64 0, %x
  %and = and i64 %sub, 1
  %add = add i64 %sub, %and
  ret i64 %add
}

