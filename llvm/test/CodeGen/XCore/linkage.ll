; RUN: llc < %s -march=xcore | FileCheck %s

; CHECK: .weak fd
define weak void @fd() {
  call void @fr(ptr @gd, ptr @gr)
  ret void
}

; CHECK-NOT: .hidden test_hidden
declare hidden void @test_hidden_declaration()
define hidden void @test_hidden() {
  call void @test_hidden_declaration()
  unreachable
}

; CHECK-NOT: .protected
define protected void @test_protected() {
  unreachable
}

; CHECK: .globl array.globound
; CHECK: .set array.globound, 2
; CHECK: .weak array.globound
; CHECK: .globl array
; CHECK: .weak array
@array = weak global [2 x i32] zeroinitializer

; CHECK: .globl ac.globound
; CHECK: .set ac.globound, 2
; CHECK: .weak ac.globound
; CHECK: .globl ac
; CHECK: .weak ac
@ac = common global [2 x i32] zeroinitializer

; CHECK: .globl gd
; CHECK: .weak gd
@gd = weak global i32 0

; CHECK: .globl gc
; CHECK: .weak gc
@gc = common global i32 0

; CHECK-NOT: .hidden test_hidden_declaration

; CHECK: .weak fr
declare extern_weak void @fr(ptr, ptr)

; CHECK: .weak gr
@gr = extern_weak global i32
