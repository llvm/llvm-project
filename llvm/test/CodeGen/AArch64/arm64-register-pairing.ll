; RUN: llc -mtriple=arm64-apple-ios < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu < %s | FileCheck -check-prefix CHECK-NOTMACHO %s
;
; rdar://14075006

define void @odd() nounwind {
; CHECK-LABEL: odd:
; CHECK: stp d15, d14, [sp, #-144]!
; CHECK: stp d13, d12, [sp, #16]
; CHECK: stp d11, d10, [sp, #32]
; CHECK: stp d9, d8, [sp, #48]
; CHECK: stp x28, x27, [sp, #64]
; CHECK: stp x26, x25, [sp, #80]
; CHECK: stp x24, x23, [sp, #96]
; CHECK: stp x22, x21, [sp, #112]
; CHECK: stp x20, x19, [sp, #128]
; CHECK: mov x0, #42
; CHECK: ldp x20, x19, [sp, #128]
; CHECK: ldp x22, x21, [sp, #112]
; CHECK: ldp x24, x23, [sp, #96]
; CHECK: ldp x26, x25, [sp, #80]
; CHECK: ldp x28, x27, [sp, #64]
; CHECK: ldp d9, d8, [sp, #48]
; CHECK: ldp d11, d10, [sp, #32]
; CHECK: ldp d13, d12, [sp, #16]
; CHECK: ldp d15, d14, [sp], #144

; CHECK-NOTMACHO-LABEL: odd:
; CHECK-NOTMACHO: stp d14, d12, [sp, #-80]!
; CHECK-NOTMACHO: stp d10, d8, [sp, #16]
; CHECK-NOTMACHO: str x27, [sp, #32]
; CHECK-NOTMACHO: stp x25, x23, [sp, #48]
; CHECK-NOTMACHO: stp x21, x19, [sp, #64]
; CHECK-NOTMACHO: mov x0, #42
; CHECK-NOTMACHO: ldp x21, x19, [sp, #64]
; CHECK-NOTMACHO: ldp x25, x23, [sp, #48]
; CHECK-NOTMACHO: ldr x27, [sp, #32]
; CHECK-NOTMACHO: ldp d10, d8, [sp, #16]
; CHECK-NOTMACHO: ldp d14, d12, [sp], #80
  call void asm sideeffect "mov x0, #42", "~{x0},~{x19},~{x21},~{x23},~{x25},~{x27},~{d8},~{d10},~{d12},~{d14}"() nounwind
  ret void
}

define void @even() nounwind {
; CHECK-LABEL: even:
; CHECK: stp d15, d14, [sp, #-144]!
; CHECK: stp d13, d12, [sp, #16]
; CHECK: stp d11, d10, [sp, #32]
; CHECK: stp d9, d8, [sp, #48]
; CHECK: stp x28, x27, [sp, #64]
; CHECK: stp x26, x25, [sp, #80]
; CHECK: stp x24, x23, [sp, #96]
; CHECK: stp x22, x21, [sp, #112]
; CHECK: stp x20, x19, [sp, #128]
; CHECK: mov x0, #42
; CHECK: ldp x20, x19, [sp, #128]
; CHECK: ldp x22, x21, [sp, #112]
; CHECK: ldp x24, x23, [sp, #96]
; CHECK: ldp x26, x25, [sp, #80]
; CHECK: ldp x28, x27, [sp, #64]
; CHECK: ldp d9, d8, [sp, #48]
; CHECK: ldp d11, d10, [sp, #32]
; CHECK: ldp d13, d12, [sp, #16]
; CHECK: ldp d15, d14, [sp], #144

; CHECK-NOTMACHO-LABEL: even:
; CHECK-NOTMACHO: stp d15, d13, [sp, #-80]!
; CHECK-NOTMACHO: stp d11, d9, [sp, #16]
; CHECK-NOTMACHO: str x28, [sp, #32]
; CHECK-NOTMACHO: stp x26, x24, [sp, #48]
; CHECK-NOTMACHO: stp x22, x20, [sp, #64]
; CHECK-NOTMACHO: mov x0, #42
; CHECK-NOTMACHO: ldp x22, x20, [sp, #64]
; CHECK-NOTMACHO: ldp x26, x24, [sp, #48]
; CHECK-NOTMACHO: ldr x28, [sp, #32]
; CHECK-NOTMACHO: ldp d11, d9, [sp, #16]
; CHECK-NOTMACHO: ldp d15, d13, [sp], #80
  call void asm sideeffect "mov x0, #42", "~{x0},~{x20},~{x22},~{x24},~{x26},~{x28},~{d9},~{d11},~{d13},~{d15}"() nounwind
  ret void
}
