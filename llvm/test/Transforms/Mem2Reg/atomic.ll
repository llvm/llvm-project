; RUN: opt -passes=mem2reg < %s -S | FileCheck %s

; mem2reg is allowed with arbitrary atomic operations (although we only support
; it for atomic load and store at the moment).
define i32 @test1(i32 %x) {
; CHECK-LABEL: @test1(
; CHECK: ret i32 %x
  %a = alloca i32
  store atomic i32 %x, ptr %a seq_cst, align 4
  %r = load atomic i32, ptr %a seq_cst, align 4
  ret i32 %r
}
