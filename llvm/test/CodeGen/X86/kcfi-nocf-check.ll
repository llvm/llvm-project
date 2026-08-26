; RUN: llc -mtriple=x86_64-unknown-unknown -x86-indirect-branch-tracking < %s | FileCheck %s

; CHECK-LABEL: __cfi_cf_check_func:
; CHECK:       movl	$12345678, %eax
define void @cf_check_func() !kcfi_type !2 {
; CHECK-LABEL: cf_check_func:
; CHECK:       endbr64
; CHECK:       retq
entry:
  ret void
}

; CHECK-NOT:   __cfi_notype_cf_check_func:
; CHECK-NOT:   movl
define void @notype_cf_check_func() {
; CHECK-LABEL: notype_cf_check_func:
; CHECK:       endbr64
; CHECK:       retq
entry:
  ret void
}

; CHECK-NOT:   __cfi_nocf_check_func:
; CHECK-NOT:   movl
define void @nocf_check_func() #0 !kcfi_type !2 {
; CHECK-LABEL: nocf_check_func:
; CHECK-NOT:   endbr64
; CHECK:       retq
entry:
  ret void
}

attributes #0 = { nocf_check }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 4, !"kcfi", i32 1}
!2 = !{i32 12345678}
