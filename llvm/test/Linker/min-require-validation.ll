; Zeroing an absent Min flag must update the lookup used by Require validation.
; Use an unrelated flag so both inputs have module-flags metadata: this exercises
; the normal merging path, independently of flagless-input initialization.
;
; Disable the final verifier to ensure the linker itself diagnoses the unmet
; requirement instead of returning success with invalid module metadata.
; RUN: split-file %s %t
; RUN: llvm-link %t/required.ll -S -o - \
; RUN:   | FileCheck %s --check-prefix=VALID
; RUN: not llvm-link -disable-verify %t/required.ll %t/unrelated.ll -S -o - 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ERROR
; RUN: not llvm-link -disable-verify %t/unrelated.ll %t/required.ll -S -o - 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ERROR
;
; VALID: !{{[0-9]+}} = !{i32 8, !"test.min", i32 7}
; ERROR: error: linking module flags 'test.min': does not have the required value

;--- required.ll
define void @required() {
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"test.min", i32 7}
!1 = !{i32 3, !"test.require", !2}
!2 = !{!"test.min", i32 7}

;--- unrelated.ll
define void @unrelated() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"test.unrelated", i32 1}
