; RUN: llc -mtriple=aarch64-- -verify-machineinstrs < %s | FileCheck %s

; CHECK:          .p2align 2
; CHECK-NOT:        nop
; CHECK:          .word   12345678
; CHECK-LABEL:    f1:
define void @f1(ptr noundef %x) !kcfi_type !1 {
; CHECK:            ldur    w16, [x0, #-4]
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; CHECK:          .p2align 2
; CHECK-NOT:       .word
; CHECK-NOT:        nop
; CHECK-LABEL:    f2:
define void @f2(ptr noundef %x) {
; CHECK:            ldur    w16, [x0, #-4]
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; CHECK:          .p2align 2
; CHECK:          .word   12345678
; CHECK-COUNT-11:   nop
; CHECK-LABEL:    f3:
define void @f3(ptr noundef %x) #0 !kcfi_type !1 {
; CHECK:            ldur    w16, [x0, #-48]
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; CHECK:          .p2align 2
; CHECK-NOT:      .word
; CHECK-COUNT-11:   nop
; CHECK-LABEL:    f4:
define void @f4(ptr noundef %x) #0 {
; CHECK:            ldur    w16, [x0, #-48]
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

attributes #0 = { "patchable-function-prefix"="11" }

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 12345678}
