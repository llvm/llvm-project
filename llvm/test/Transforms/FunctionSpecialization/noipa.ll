; RUN: opt -passes='ipsccp<func-spec>' -force-specialization -S < %s | FileCheck %s

; Function specialisation should not activate for `noipa` functions.

define i32 @foo(i32 %x, i32 %y) noipa {
  %r2 = add i32 %x, %y
  ret i32 %r2
}

; CHECK:      call i32 @foo(
; CHECK-NOT:  define internal i32 @foo.specialized.1(
define i32 @test() {
  %r = call i32 @foo(i32 0, i32 5)
  ret i32 %r
}
