; RUN: opt < %s -passes=correlated-propagation -S | FileCheck %s

; Tests: test_nop and tests 1 through 6 are taken from udiv.ll
; with udiv replaced by lshr (plus some tweaks).
; In those tests the lshr instruction has no users.

; CHECK-LABEL: @test_nop
define void @test_nop(i32 %n) {
; CHECK: lshr i32 %n, 2
  %shr = lshr i32 %n, 2
  ret void
}

; CHECK-LABEL: @test1(
define void @test1(i32 %n) {
entry:
  %cmp = icmp ule i32 %n, 65535
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: lshr i16
  %shr = lshr i32 %n, 2
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test2(
define void @test2(i32 %n) {
entry:
  %cmp = icmp ule i32 %n, 65536
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: lshr i32
  %shr = lshr i32 %n, 2
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test3(
define void @test3(i32 %m, i32 %n) {
entry:
  %cmp1 = icmp ult i32 %m, 65535
  %cmp2 = icmp ult i32 %n, 65535
  %cmp = and i1 %cmp1, %cmp2
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: lshr i16
  %shr = lshr i32 %m, %n
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test4(
define void @test4(i32 %m, i32 %n) {
entry:
  %cmp1 = icmp ult i32 %m, 65535
  %cmp2 = icmp ule i32 %n, 65536
  %cmp = and i1 %cmp1, %cmp2
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: lshr i32
  %shr = lshr i32 %m, %n
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: @test5
define void @test5(i32 %n) {
  %trunc = and i32 %n, 65535
  ; CHECK: lshr i16
  %shr = lshr i32 %trunc, 2
  ret void
}

; CHECK-LABEL: @test6
define void @test6(i32 %n) {
entry:
  %cmp = icmp ule i32 %n, 255
  br i1 %cmp, label %bb, label %exit

bb:
; CHECK: lshr i8
  %shr = lshr i32 %n, 2
  br label %exit

exit:
  ret void
}

; The tests below check whether the narrowing occures only if the appropriate
; `trunc` instructions follow.
;
; Just as in udiv.ll, additional zext and trunc instructions appear. 
; They are eventually recombined by InstCombinePass 
; that follows in the pipeline.

; CHECK-LABEL: @trunc_test1
; CHECK-NEXT: [[A1:%.*]] = lshr i32 [[A:%.*]], 16
; CHECK-NEXT: [[B1:%.*]] = and i32 [[B:%.*]], 65535
; CHECK-NEXT: [[A2:%.*]] = trunc i32 [[A1]] to i16
; CHECK-NEXT: [[B2:%.*]] = trunc i32 [[B1]] to i16
; CHECK-NEXT: [[C1:%.*]] = lshr i16 [[A2]], [[B2]]
; CHECK-NEXT: [[C2:%.*]] = zext i16 [[C1]] to i32
; CHECK-NEXT: [[C3:%.*]] = trunc i32 [[C2]] to i16
; CHECK-NEXT: ret i16 [[C3]]

define i16 @trunc_test1(i32 %a, i32 %b) {
  %a.eff.trunc = lshr i32 %a, 16
  %b.eff.trunc = and i32 %b, 65535
  %c = lshr i32 %a.eff.trunc, %b.eff.trunc
  %c.trunc = trunc i32 %c to i16
  ret i16 %c.trunc
}

; CHECK-LABEL: @trunc_test2
; CHECK-NEXT: [[A1:%.*]] = zext i16 [[A:%.*]] to i32
; CHECK-NEXT: [[A2:%.*]] = trunc i32 [[A1]] to i16
; CHECK-NEXT: [[C1:%.*]] = lshr i16 [[A2]], 2
; CHECK-NEXT: [[C2:%.*]] = zext i16 [[C1]] to i32
; CHECK-NEXT: [[C3:%.*]] = trunc i32 [[C2]] to i16
; CHECK-NEXT: ret i16 [[C3]]

define i16 @trunc_test2(i16 %a) {
  %a.ext = zext i16 %a to i32
  %c = lshr i32 %a.ext, 2
  %c.trunc = trunc i32 %c to i16
  ret i16 %c.trunc
}

; CHECK-LABEL: @trunc_test3
; CHECK-NEXT: [[A1:%.*]] = zext i16 [[A:%.*]] to i32
; CHECK-NEXT: [[B:%.*]] = lshr i32 [[A1]], 2
; CHECK-NEXT: [[C0:%.*]] = add nuw nsw i32 [[B]], 123
; CHECK-NEXT: [[C1:%.*]] = trunc i32 [[C0]] to i16
; CHECK-NEXT: ret i16 [[C1]]

define i16 @trunc_test3(i16 %a) {
  %a.ext = zext i16 %a to i32
  %b = lshr i32 %a.ext, 2
  %c = add i32 %b, 123
  %c.trunc = trunc i32 %c to i16
  ret i16 %c.trunc
}

; CHECK-LABEL: @trunc_test4
; CHECK-NEXT: [[A1:%.*]] = udiv i32 [[A:%.*]], 17000000
; CHECK-NEXT: [[B0:%.*]] = trunc i32 [[A1]] to i16
; CHECK-NEXT: [[B1:%.*]] = lshr i16 [[B0]], 2
; CHECK-NEXT: [[B2:%.*]] = zext i16 [[B1]] to i32
; CHECK-NEXT: [[C1:%.*]] = trunc i32 [[B2]] to i16
; CHECK-NEXT: [[C2:%.*]] = trunc i32 [[B2]] to i8
; CHECK-NEXT: ret i16 [[C1]]

define i16 @trunc_test4(i32 %a) {
  %a.eff.trunc = udiv i32 %a, 17000000  ; larger than 2^24
  %b = lshr i32 %a.eff.trunc, 2 
  %b.trunc.1 = trunc i32 %b to i16
  %b.trunc.2 = trunc i32 %b to i8
  ret i16 %b.trunc.1
}

; CHECK-LABEL: @trunc_test5
; CHECK-NEXT: [[A1:%.*]] = udiv i32 [[A:%.*]], 17000000
; CHECK-NEXT: [[B:%.*]] = lshr i32 [[A1]], 2
; CHECK-NEXT: [[B1:%.*]] = trunc i32 [[B]] to i16
; CHECK-NEXT: [[B2:%.*]] = trunc i32 [[B]] to i8
; CHECK-NEXT: [[C:%.*]] = add nuw nsw i32 [[B]], 123
; CHECK-NEXT: ret i32 [[C]]

define i32 @trunc_test5(i32 %a) {
  %a.eff.trunc = udiv i32 %a, 17000000  ; larger than 2^24
  %b = lshr i32 %a.eff.trunc, 2 
  %b.trunc.1 = trunc i32 %b to i16
  %b.trunc.2 = trunc i32 %b to i8
  %c = add i32 %b, 123
  ret i32 %c
}

; Test cases where lshr simplifies to zero or poison.

; CHECK-LABEL: @zero_test1
; CHECK-NEXT: [[C:%.*]] = add i32 poison, 123
; CHECK-NEXT: ret i32 [[C]]
  
define i32 @zero_test1(i32 %a) {
  %b = lshr i32 %a, 32
  %c = add i32 %b, 123
  ret i32 %c
}

; CHECK-LABEL: @zero_test2
; CHECK-NEXT: [[B1:%.*]] = add nuw nsw i32 [[B:%.*]], 50
; CHECK-NEXT: [[D:%.*]] = add i32 poison, 123
; CHECK-NEXT: ret i32 [[D]]

define i32 @zero_test2(i32 %a, i32 %b) {
  %b.large = add nuw nsw i32 %b, 50
  %c = lshr i32 %a, %b.large
  %d = add i32 %c, 123
  ret i32 %d
}

; CHECK-LABEL: @zero_test3
; CHECK-NEXT: [[A1:%.*]] = lshr i32 [[A:%.*]], 16
; CHECK-NEXT: [[B1:%.*]] = add nuw nsw i32 [[B:%.*]], 20
; CHECK-NEXT: [[D:%.*]] = add nuw nsw i32 0, 123
; CHECK-NEXT: ret i32 123

define i32 @zero_test3(i32 %a, i32 %b) {
  %a.small = lshr i32 %a, 16
  %b.large = add nuw nsw i32 %b, 20
  %c = lshr i32 %a.small, %b.large
  %d = add i32 %c, 123
  ret i32 %d
}

; CHECK-LABEL: @zero_test4
; CHECK-NEXT: [[A1:%.*]] = lshr i32 [[A:%.*]], 16
; CHECK-NEXT: [[B1:%.*]] = add nuw nsw i32 [[B:%.*]], 20
; CHECK-NEXT: [[C:%.*]] = lshr exact i32 [[A1]], [[B1]]
; CHECK-NEXT: [[D:%.*]] = add nuw nsw i32 [[C]], 123
; CHECK-NEXT: ret i32 123

define i32 @zero_test4(i32 %a, i32 %b) {
  %a.small = lshr i32 %a, 16
  %b.large = add nuw nsw i32 %b, 20
  %c = lshr exact i32 %a.small, %b.large
  %d = add i32 %c, 123
  ret i32 %d
}
