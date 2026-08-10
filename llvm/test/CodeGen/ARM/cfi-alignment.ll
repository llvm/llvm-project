; RUN: llc -mtriple=thumbv7k-apple-watchos7.0 -o - %s | FileCheck %s

; Since d11 doesn't get pushed with the aligned registers, its frameindex
; shouldn't be modified to say it has been.

define void @foo() {
; CHECK-LABEL: foo:
; CHECK: push {r7, lr}
; CHECK: .cfi_offset r7, -8
; CHECK: vpush {d11}
; CHECK: vpush {d8, d9}
; CHECK: .cfi_offset d11, -16
; CHECK: .cfi_offset d9, -24
; CHECK: .cfi_offset d8, -32
  call void asm sideeffect "", "~{d8},~{d9},~{d11}"()
  call void @bar()
  ret void
}

define void @variadic_foo(i8, ...) {
; CHECK-LABEL: variadic_foo:
; CHECK: sub sp, #12
; CHECK: push {r7, lr}
; CHECK: .cfi_offset r7, -20
; CHECK: sub sp, #4
; CHECK: vpush {d11}
; CHECK: vpush {d8, d9}
; CHECK: .cfi_offset d11, -32
; CHECK: .cfi_offset d9, -40
; CHECK: .cfi_offset d8, -48
  call void asm sideeffect "", "~{d8},~{d9},~{d11}"()
  call void @llvm.va_start(ptr null)
  call void @bar()
  ret void
}

define void @test_maintain_stack_align() {
; CHECK-LABEL: test_maintain_stack_align:
; CHECK: push {r7, lr}
; CHECK: vpush {d8, d9}
; CHECK: sub sp, #8
  call void asm sideeffect "", "~{d8},~{d9}"()
  call void @bar()
  ret void
}

declare void @bar()
declare void @llvm.va_start(ptr) nounwind
