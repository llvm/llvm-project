; RUN: opt -S -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

declare i32 @f(i32 %val)

; Check that eliminating cases with unreachable branches keeps
; prof branch_weights metadata consistent with switch instruction.
define i32 @test_switch_to_unreachable(i32 %val) {
; CHECK-LABEL: test_switch_to_unreachable
; CHECK: switch
; CHECK-NOT: i32 0, label %on0
  switch i32 %val, label %otherwise [
    i32 0, label %on0
    i32 1, label %on1
    i32 2, label %on2
  ], !prof !{!"branch_weights", i32 99, i32 0, i32 1, i32 2}
; CHECK: !prof ![[MD0:[0-9]+]]

otherwise:
  %result = call i32 @f(i32 -1)
  ret i32 %result

on0:
  unreachable
  ret i32 125

on1:
  %result1 = call i32 @f(i32 -2)
  ret i32 %result1

on2:
  %result2 = call i32 @f(i32 -3)
  ret i32 %result2
}

; CHECK: ![[MD0]] = !{!"branch_weights", i32 99, i32 2, i32 1}
