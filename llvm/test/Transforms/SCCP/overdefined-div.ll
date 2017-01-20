; RUN: opt < %s -sccp -S | FileCheck %s

; Test that SCCP has basic knowledge of when div can nuke overdefined values.

; 0 / X = 0 even if X is overdefined.
; CHECK-LABEL: test1
; CHECK-NEXT: ret i32 0
define i32 @test1(i32 %foo) {
  %tinkywinky = udiv i32 0, %foo
  ret i32 %tinkywinky
}

; CHECK-LABEL: test2
; CHECK-NEXT: ret i32 0
define i32 @test2(i32 %foo) {
  %tinkywinky = sdiv i32 0, %foo
  ret i32 %tinkywinky
}

; CHECK-LABEL: test3
; CHECK: ret i32 %tinkywinky
define i32 @test3(i32 %foo) {
  %tinkywinky = udiv i32 %foo, 0
  ret i32 %tinkywinky
}

; CHECK-LABEL: test4
; CHECK: ret i32 %tinkywinky
define i32 @test4(i32 %foo) {
  %tinkywinky = sdiv i32 %foo, 0
  ret i32 %tinkywinky
}
