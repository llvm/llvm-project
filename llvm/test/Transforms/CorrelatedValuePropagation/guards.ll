; RUN: opt -correlated-propagation -S < %s | FileCheck %s

declare void @llvm.experimental.guard(i1,...)

define i1 @test1(i32 %a) {
; CHECK-LABEL: @test1(
; CHECK: %alive = icmp eq i32 %a, 8
; CHECK-NEXT: %result = or i1 false, %alive
  %cmp = icmp ult i32 %a, 16
  call void(i1,...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  %dead = icmp eq i32 %a, 16
  %alive = icmp eq i32 %a, 8
  %result = or i1 %dead, %alive
  ret i1 %result
}

define i1 @test2(i32 %a) {
; CHECK-LABEL: @test2(
; CHECK: continue:
; CHECK-NEXT: %alive = icmp eq i32 %a, 8
; CHECK-NEXT: %result = or i1 false, %alive
  %cmp = icmp ult i32 %a, 16
  call void(i1,...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  br label %continue

continue:
  %dead = icmp eq i32 %a, 16
  %alive = icmp eq i32 %a, 8
  %result = or i1 %dead, %alive
  ret i1 %result
}

define i1 @test3(i32 %a, i1 %flag) {
; CHECK-LABEL: @test3(
; CHECK: continue:
; CHECK-NEXT: %alive.1 = icmp eq i32 %a, 16
; CHECK-NEXT: %alive.2 = icmp eq i32 %a, 8
; CHECK-NEXT: %result = or i1 %alive.1, %alive.2
  br i1 %flag, label %true, label %false

true:
  %cmp = icmp ult i32 %a, 16
  call void(i1,...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  br label %continue

false:
  br label %continue

continue:
  %alive.1 = icmp eq i32 %a, 16
  %alive.2 = icmp eq i32 %a, 8
  %result = or i1 %alive.1, %alive.2
  ret i1 %result
}

define i1 @test4(i32 %a, i1 %flag) {
; CHECK-LABEL: @test4(
; CHECK: continue:
; CHECK-NEXT: %alive = icmp eq i32 %a, 12
; CHECK-NEXT: %result = or i1 false, %alive
  br i1 %flag, label %true, label %false

true:
  %cmp.t = icmp ult i32 %a, 16
  call void(i1,...) @llvm.experimental.guard(i1 %cmp.t) [ "deopt"() ]
  br label %continue

false:
  %cmp.f = icmp ult i32 %a, 12
  call void(i1,...) @llvm.experimental.guard(i1 %cmp.f) [ "deopt"() ]
  br label %continue

continue:
  %dead = icmp eq i32 %a, 16
  %alive = icmp eq i32 %a, 12
  %result = or i1 %dead, %alive
  ret i1 %result
}

define i1 @test5(i32 %a) {
; CHECK-LABEL: @test5(
; CHECK: continue:
; CHECK-NEXT: %alive = icmp eq i32 %a.plus.8, 16
; CHECK-NEXT: %result = or i1 false, %alive
  %cmp = icmp ult i32 %a, 16
  call void(i1,...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  %a.plus.8 = add i32 %a, 8
  br label %continue

continue:
  %dead = icmp eq i32 %a.plus.8, 24
  %alive = icmp eq i32 %a.plus.8, 16
  %result = or i1 %dead, %alive
  ret i1 %result
}

; Check that we handle the case when the guard is the very first instruction in
; a basic block.
define i1 @test6(i32 %a) {
; CHECK-LABEL: @test6(
; CHECK: %alive = icmp eq i32 %a, 8
; CHECK-NEXT: %result = or i1 false, %alive
  %cmp = icmp ult i32 %a, 16
  br label %continue

continue:
  call void(i1,...) @llvm.experimental.guard(i1 %cmp) [ "deopt"() ]
  %dead = icmp eq i32 %a, 16
  %alive = icmp eq i32 %a, 8
  %result = or i1 %dead, %alive
  ret i1 %result
}
