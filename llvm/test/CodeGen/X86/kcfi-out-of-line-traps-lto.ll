; RUN: split-file %s %t
; RUN: llvm-link %t/ool.ll %t/inline.ll | llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs | FileCheck %s --check-prefix=INLINE
; RUN: llvm-link %t/inline.ll %t/ool.ll | llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs | FileCheck %s --check-prefix=INLINE-REV
; RUN: llvm-link %t/ool.ll %t/ool2.ll | llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs | FileCheck %s --check-prefix=OOL
; RUN: llvm-link %t/ool.ll %t/nokcfi.ll | llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs | FileCheck %s --check-prefix=NOKCFI
; RUN: llvm-link %t/nokcfi.ll %t/ool.ll | llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs | FileCheck %s --check-prefix=NOKCFI

;; Verify kcfi-out-of-line-traps module flag semantics:
;; - KCFI opt-in emits Max (7, value 1) and KCFI inline emits Override (4, value 0),
;;   so linking any KCFI-inline module forces inline traps regardless of link order.
;; - Non-KCFI modules omit the flag so linking with uninstrumented bitcode preserves
;;   out-of-line traps in KCFI modules.

;--- ool.ll
define void @f_ool(ptr noundef %x) {
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 7, !"kcfi-out-of-line-traps", i32 1}

;--- ool2.ll
define void @f_ool2(ptr noundef %x) {
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 7, !"kcfi-out-of-line-traps", i32 1}

; OOL-LABEL: f_ool:
; OOL:         jne [[TRAP1:.LBB[0-9_]+]]
; OOL:         callq *%rdi
; OOL:       [[TRAP1]]:
; OOL:         ud2
; OOL-LABEL: f_ool2:
; OOL:         jne [[TRAP2:.LBB[0-9_]+]]
; OOL:         callq *%rdi
; OOL:       [[TRAP2]]:
; OOL:         ud2

;--- inline.ll
define void @f_inline(ptr noundef %x) {
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 4, !"kcfi-out-of-line-traps", i32 0}

; INLINE-LABEL: f_ool:
; INLINE:         je [[PASS1:.Ltmp[0-9]+]]
; INLINE-NEXT:  .Ltmp{{[0-9]+}}:
; INLINE-NEXT:    ud2
; INLINE:       [[PASS1]]:
; INLINE-NEXT:    callq *%rdi
; INLINE-LABEL: f_inline:
; INLINE:         je [[PASS2:.Ltmp[0-9]+]]
; INLINE-NEXT:  .Ltmp{{[0-9]+}}:
; INLINE-NEXT:    ud2
; INLINE:       [[PASS2]]:
; INLINE-NEXT:    callq *%rdi

; INLINE-REV-LABEL: f_inline:
; INLINE-REV:         je [[PASS3:.Ltmp[0-9]+]]
; INLINE-REV-NEXT:  .Ltmp{{[0-9]+}}:
; INLINE-REV-NEXT:    ud2
; INLINE-REV:       [[PASS3]]:
; INLINE-REV-NEXT:    callq *%rdi
; INLINE-REV-LABEL: f_ool:
; INLINE-REV:         je [[PASS4:.Ltmp[0-9]+]]
; INLINE-REV-NEXT:  .Ltmp{{[0-9]+}}:
; INLINE-REV-NEXT:    ud2
; INLINE-REV:       [[PASS4]]:
; INLINE-REV-NEXT:    callq *%rdi

;--- nokcfi.ll
define void @f_nokcfi() {
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"wchar_size", i32 4}

; NOKCFI-LABEL: f_ool:
; NOKCFI:         jne [[TRAP:.LBB[0-9_]+]]
; NOKCFI:         callq *%rdi
; NOKCFI:       [[TRAP]]:
; NOKCFI:         ud2
