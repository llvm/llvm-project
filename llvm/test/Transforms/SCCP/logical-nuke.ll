; RUN: opt < %s -sccp -S | FileCheck %s

; Test that SCCP has basic knowledge of when and/or/mul nuke overdefined values.

; CHECK-LABEL: test
; CHECK: ret i32 0
 define i32 @test(i32 %X) {
  %Y = and i32 %X, 0
  ret i32 %Y
}

; CHECK-LABEL: test2
; CHECK: ret i32 -1
define i32 @test2(i32 %X) {
  %Y = or i32 -1, %X
  ret i32 %Y
}

; CHECK-LABEL: test3
; CHECK: ret i32 0
define i32 @test3(i32 %X) {
  %Y = and i32 undef, %X
  ret i32 %Y
}

; CHECK-LABEL: test4
; CHECK: ret i32 -1
define i32 @test4(i32 %X) {
  %Y = or i32 %X, undef
  ret i32 %Y
}

; X * 0 = 0 even if X is overdefined.
; CHECK-LABEL: test5
; CHECK: ret i32 0
define i32 @test5(i32 %foo) {
  %patatino = mul i32 %foo, 0
  ret i32 %patatino
}
