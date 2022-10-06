; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s

; CHECK:          .p2align 4, 0x90
; CHECK-LABEL:    __cfi_f1:
; CHECK-COUNT-11:   nop
; CHECK-NEXT:       movl $12345678, %eax
; CHECK-LABEL:    .Lcfi_func_end0:
; CHECK-NEXT:     .size   __cfi_f1, .Lcfi_func_end0-__cfi_f1
; CHECK-LABEL:    f1:
define void @f1(ptr noundef %x) !kcfi_type !1 {
; CHECK:            addl -4(%r{{..}}), %r10d
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; CHECK:          .p2align 4, 0x90
; CHECK-NOT:      __cfi_f2:
; CHECK-NOT:        nop
; CHECK-LABEL:    f2:
define void @f2(ptr noundef %x) {
; CHECK:            addl -4(%r{{..}}), %r10d
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; CHECK:          .p2align 4, 0x90
; CHECK-LABEL:    __cfi_f3:
; CHECK-NOT:        nop
; CHECK-NEXT:       movl $12345678, %eax
; CHECK-COUNT-11:   nop
; CHECK-LABEL:    f3:
define void @f3(ptr noundef %x) #0 !kcfi_type !1 {
; CHECK:            addl -15(%r{{..}}), %r10d
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; CHECK:          .p2align 4, 0x90
; CHECK-NOT:      __cfi_f4:
; CHECK-COUNT-16:   nop
; CHECK-LABEL:    f4:
define void @f4(ptr noundef %x) #0 {
; CHECK:            addl -15(%r{{..}}), %r10d
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

attributes #0 = { "patchable-function-prefix"="11" }

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 12345678}
