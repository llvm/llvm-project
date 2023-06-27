; RUN: llc -mtriple=riscv64 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,NOC
; RUN: llc -mtriple=riscv64 -mattr=+c -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,C

; NOC:            .p2align 2
; C:              .p2align 1
; CHECK-NOT:        nop
; CHECK:          .word   12345678
; CHECK-LABEL:    f1:
define void @f1(ptr noundef %x) !kcfi_type !1 {
; CHECK:            lw      t1, -4(a0)
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; NOC:            .p2align 2
; C:              .p2align 1
; CHECK-NOT:       .word
; CHECK-NOT:        nop
; CHECK-LABEL:    f2:
define void @f2(ptr noundef %x) {
; CHECK:            lw      t1, -4(a0)
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; NOC:            .p2align 2
; C:              .p2align 1
; CHECK:          .word   12345678
; CHECK-COUNT-11:   nop
; CHECK-LABEL:    f3:
define void @f3(ptr noundef %x) #0 !kcfi_type !1 {
; NOC:              lw      t1, -48(a0)
; C:                lw      t1, -26(a0)
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; NOC:            .p2align 2
; C:              .p2align 1
; CHECK-NOT:      .word
; CHECK-COUNT-11:   nop
; CHECK-LABEL:    f4:
define void @f4(ptr noundef %x) #0 {
; NOC:            lw      t1, -48(a0)
; C:              lw      t1, -26(a0)
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

attributes #0 = { "patchable-function-prefix"="11" }

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 12345678}
