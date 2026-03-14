; RUN: llc -verify-machineinstrs -mtriple=arm64-apple-ios7.0 -o - %s | FileCheck %s
@var = global ptr zeroinitializer

declare void @bar()

define void @foo() {
; CHECK-LABEL: foo:
       %func = load ptr, ptr @var

       ; Calling a function encourages @foo to use a callee-saved register,
       ; which makes it a natural choice for the tail call itself. But we don't
       ; want that: the final "br xN" has to use a temporary or argument
       ; register.
       call void @bar()

       tail call void %func()
; CHECK: br {{x([0-79]|1[0-8])}}
       ret void
}

; No matter how tempting it is, LLVM should not use x30 since that'll be
; restored to its incoming value before the "br".
define void @test_x30_tail() {
; CHECK-LABEL: test_x30_tail:
; CHECK: mov [[DEST:x[0-9]+]], x30
; CHECK: br [[DEST]]
  %addr = call ptr @llvm.returnaddress(i32 0)
  tail call void %addr()
  ret void
}

declare ptr @llvm.returnaddress(i32)
