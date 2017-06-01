; RUN: opt -correlated-propagation -S < %s | FileCheck %s
; Checks that we don't crash on conflicting facts about a value 
; (i.e. unreachable code)

; Test that we can handle conflict edge facts
define i8 @test(i8 %a) {
; CHECK-LABEL: @test
  %cmp1 = icmp eq i8 %a, 5
  br i1 %cmp1, label %next, label %exit
next:
  %cmp2 = icmp eq i8 %a, 3
; CHECK: br i1 false, label %dead, label %exit
  br i1 %cmp2, label %dead, label %exit
dead:
; CHECK-LABEL: dead:
; CHECK: ret i8 5
; NOTE: undef, or 3 would be equal valid
  ret i8 %a
exit:
  ret i8 0
}

declare void @llvm.assume(i1)

; Test that we can handle conflicting assume vs edge facts
define i8 @test2(i8 %a) {
; CHECK-LABEL: @test2
  %cmp1 = icmp eq i8 %a, 5
  call void @llvm.assume(i1 %cmp1)
  %cmp2 = icmp eq i8 %a, 3
; CHECK: br i1 false, label %dead, label %exit
  br i1 %cmp2, label %dead, label %exit
dead:
  ret i8 %a
exit:
  ret i8 0
}

define i8 @test3(i8 %a) {
; CHECK-LABEL: @test3
  %cmp1 = icmp eq i8 %a, 5
  br i1 %cmp1, label %dead, label %exit
dead:
  %cmp2 = icmp eq i8 %a, 3
; CHECK: call void @llvm.assume(i1 false)
  call void @llvm.assume(i1 %cmp2)
  ret i8 %a
exit:
  ret i8 0
}
