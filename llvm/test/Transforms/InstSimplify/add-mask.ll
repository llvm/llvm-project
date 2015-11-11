; RUN: opt -S -instsimplify < %s | FileCheck %s

define i1 @test(i32 %a) {
; CHECK-LABEL: @test
; CHECK: ret i1 false
  %rhs = add i32 %a, -1
  %and = and i32 %a, %rhs
  %res = icmp eq i32 %and, 1
  ret i1 %res
}

define i1 @test2(i32 %a) {
; CHECK-LABEL: @test2
; CHECK: ret i1 false
  %rhs = add i32 %a, 1
  %and = and i32 %a, %rhs
  %res = icmp eq i32 %and, 1
  ret i1 %res
}

define i1 @test3(i32 %a) {
; CHECK-LABEL: @test3
; CHECK: ret i1 false
  %rhs = add i32 %a, 7
  %and = and i32 %a, %rhs
  %res = icmp eq i32 %and, 1
  ret i1 %res
}

@B = external global i32
declare void @llvm.assume(i1)

; Known bits without a constant
define i1 @test4(i32 %a) {
; CHECK-LABEL: @test4
; CHECK: ret i1 false
  %b = load i32, i32* @B
  %b.and = and i32 %b, 1
  %b.cnd = icmp eq i32 %b.and, 1
  call void @llvm.assume(i1 %b.cnd)

  %rhs = add i32 %a, %b
  %and = and i32 %a, %rhs
  %res = icmp eq i32 %and, 1
  ret i1 %res
}

; Negative test - even number
define i1 @test5(i32 %a) {
; CHECK-LABEL: @test5
; CHECK: ret i1 %res
  %rhs = add i32 %a, 2
  %and = and i32 %a, %rhs
  %res = icmp eq i32 %and, 1
  ret i1 %res
}

define i1 @test6(i32 %a) {
; CHECK-LABEL: @test6
; CHECK: ret i1 false
  %lhs = add i32 %a, -1
  %and = and i32 %lhs, %a
  %res = icmp eq i32 %and, 1
  ret i1 %res
}
