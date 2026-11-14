; RUN: split-file %s %t
; RUN: llvm-link %t/ool.ll %t/inline.ll | llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs | FileCheck %s --check-prefix=INLINE
; RUN: llvm-link %t/ool.ll %t/ool2.ll | llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs | FileCheck %s --check-prefix=OOL

;; Verify kcfi-out-of-line-traps module flag semantics:
;; Linking an opt-in module (Module::Min) with an absent module merges to 0
;; so mixed LTO links keep inline traps, while linking two opt-in modules
;; preserves out-of-line traps.

;--- ool.ll
define void @f_ool(ptr noundef %x) {
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 8, !"kcfi-out-of-line-traps", i32 1}

;--- ool2.ll
define void @f_ool2(ptr noundef %x) {
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 8, !"kcfi-out-of-line-traps", i32 1}

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
!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}

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
