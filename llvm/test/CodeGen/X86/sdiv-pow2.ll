; RUN: llc -mtriple=i686-- < %s | FileCheck %s

; No attributes, should not use idiv
define i32 @test1(i32 inreg %x) {
entry:
  %div = sdiv i32 %x, 16
  ret i32 %div
; CHECK-LABEL: test1:
; CHECK-NOT: idivl
; CHECK: ret
}

; Has minsize (-Oz) attribute, should generate idiv
define i32 @test2(i32 inreg %x) minsize {
entry:
  %div = sdiv i32 %x, 16
  ret i32 %div
; CHECK-LABEL: test2:
; CHECK: idivl
; CHECK: ret
}

; Has optsize (-Os) attribute, should not generate idiv
define i32 @test3(i32 inreg %x) optsize {
entry:
  %div = sdiv i32 %x, 16
  ret i32 %div
; CHECK-LABEL: test3:
; CHECK-NOT: idivl
; CHECK: ret
}


