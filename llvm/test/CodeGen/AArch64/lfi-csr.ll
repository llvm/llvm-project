; RUN: llc < %s -mtriple=aarch64_lfi -filetype=obj -o /dev/null
; RUN: llc < %s -mtriple=aarch64_lfi | FileCheck %s --implicit-check-not=x25 --implicit-check-not=x26 --implicit-check-not=x27 --implicit-check-not=x28

; LFI reserves x25-x28 and x30 so a function should not save/restore them.

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

define void @test_lr_saved() {
; CHECK-LABEL: test_lr_saved:
; CHECK:       str x30, [sp, #-16]!
; CHECK:       bl g
; CHECK:       ldr x30, [sp], #16
; CHECK:       ret
  call void @g()
  ret void
}

define void @test_lr_not_allocated(ptr %p) {
; CHECK-LABEL: test_lr_not_allocated:
; CHECK-NOT:   {{[xw]30}}
; CHECK:       ret
  %v0 = load volatile i64, ptr %p
  %v1 = load volatile i64, ptr %p
  %v2 = load volatile i64, ptr %p
  %v3 = load volatile i64, ptr %p
  %v4 = load volatile i64, ptr %p
  %v5 = load volatile i64, ptr %p
  %v6 = load volatile i64, ptr %p
  %v7 = load volatile i64, ptr %p
  %v8 = load volatile i64, ptr %p
  %v9 = load volatile i64, ptr %p
  %v10 = load volatile i64, ptr %p
  %v11 = load volatile i64, ptr %p
  %v12 = load volatile i64, ptr %p
  %v13 = load volatile i64, ptr %p
  %v14 = load volatile i64, ptr %p
  %v15 = load volatile i64, ptr %p
  %v16 = load volatile i64, ptr %p
  %v17 = load volatile i64, ptr %p
  %v18 = load volatile i64, ptr %p
  %v19 = load volatile i64, ptr %p
  %v20 = load volatile i64, ptr %p
  %v21 = load volatile i64, ptr %p
  %v22 = load volatile i64, ptr %p
  %v23 = load volatile i64, ptr %p
  %v24 = load volatile i64, ptr %p
  %v25 = load volatile i64, ptr %p
  %v26 = load volatile i64, ptr %p
  %v27 = load volatile i64, ptr %p
  %v28 = load volatile i64, ptr %p
  %v29 = load volatile i64, ptr %p
  store volatile i64 %v0, ptr %p
  store volatile i64 %v1, ptr %p
  store volatile i64 %v2, ptr %p
  store volatile i64 %v3, ptr %p
  store volatile i64 %v4, ptr %p
  store volatile i64 %v5, ptr %p
  store volatile i64 %v6, ptr %p
  store volatile i64 %v7, ptr %p
  store volatile i64 %v8, ptr %p
  store volatile i64 %v9, ptr %p
  store volatile i64 %v10, ptr %p
  store volatile i64 %v11, ptr %p
  store volatile i64 %v12, ptr %p
  store volatile i64 %v13, ptr %p
  store volatile i64 %v14, ptr %p
  store volatile i64 %v15, ptr %p
  store volatile i64 %v16, ptr %p
  store volatile i64 %v17, ptr %p
  store volatile i64 %v18, ptr %p
  store volatile i64 %v19, ptr %p
  store volatile i64 %v20, ptr %p
  store volatile i64 %v21, ptr %p
  store volatile i64 %v22, ptr %p
  store volatile i64 %v23, ptr %p
  store volatile i64 %v24, ptr %p
  store volatile i64 %v25, ptr %p
  store volatile i64 %v26, ptr %p
  store volatile i64 %v27, ptr %p
  store volatile i64 %v28, ptr %p
  store volatile i64 %v29, ptr %p
  ret void
}

declare void @g()
