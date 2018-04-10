; RUN: opt < %s -globalopt -S | FileCheck %s
; PR36546

; Check that musttail callee preserves its calling convention

define i32 @test(i32 %a) {
  ; CHECK: %ca = musttail call i32 @foo(i32 %a)
  %ca = musttail call i32 @foo(i32 %a)
  ret i32 %ca
}

; CHECK-LABEL: define internal i32 @foo(i32 %a)
define internal i32 @foo(i32 %a) {
  ret i32 %a
}

; Check that musttail caller preserves its calling convention

define i32 @test2(i32 %a) {
  %ca = call i32 @foo1(i32 %a)
  ret i32 %ca
}

; CHECK-LABEL: define internal i32 @foo1(i32 %a)
define internal i32 @foo1(i32 %a) {
  ; CHECK: %ca = musttail call i32 @foo2(i32 %a)
  %ca = musttail call i32 @foo2(i32 %a)
  ret i32 %ca
}

; CHECK-LABEL: define internal i32 @foo2(i32 %a)
define internal i32 @foo2(i32 %a) {
  ret i32 %a
}
