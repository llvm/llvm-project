; RUN: llc < %s -mtriple=aarch64_lfi -filetype=obj -o /dev/null
; RUN: llc < %s -mtriple=aarch64_lfi | FileCheck %s --implicit-check-not=x25 --implicit-check-not=x26 --implicit-check-not=x27 --implicit-check-not=x28

; LFI reserves x25-x28 so a function should not save/restore them.

define void @test_default() {
; CHECK-LABEL: test_default:
; CHECK:       str x19, [sp, #-16]!
; CHECK:       ldr x19, [sp], #16
  call void asm sideeffect "", "~{x19},~{x25},~{x26},~{x27},~{x28}"()
  ret void
}

define preserve_mostcc void @test_preserve_most() {
; CHECK-LABEL: test_preserve_most:
; CHECK:       str x19, [sp, #-16]!
; CHECK:       ldr x19, [sp], #16
  call void asm sideeffect "", "~{x19},~{x25},~{x26},~{x27},~{x28}"()
  ret void
}

define preserve_allcc void @test_preserve_all() {
; CHECK-LABEL: test_preserve_all:
; CHECK:       str x19, [sp, #-16]!
; CHECK:       ldr x19, [sp], #16
  call void asm sideeffect "", "~{x19},~{x25},~{x26},~{x27},~{x28}"()
  ret void
}

define swiftcc void @test_swift(ptr swiftself %s) {
; CHECK-LABEL: test_swift:
; CHECK:       str x19, [sp, #-16]!
; CHECK:       ldr x19, [sp], #16
  call void asm sideeffect "", "~{x19},~{x25},~{x26},~{x27},~{x28}"()
  ret void
}

define void @test_unwind_init() {
; CHECK-LABEL: test_unwind_init:
; CHECK:       stp x20, x19
; CHECK:       ldp x20, x19
  call void @llvm.eh.unwind.init()
  ret void
}

declare void @llvm.eh.unwind.init()
