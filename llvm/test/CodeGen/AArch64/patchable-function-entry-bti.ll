; RUN: llc -mtriple=aarch64 %s -o - | FileCheck --check-prefixes=CHECK %s

define void @f0() "patchable-function-entry"="0" "branch-target-enforcement" {
; CHECK-LABEL: f0:
; CHECK-NEXT: .Lfunc_begin0:
; CHECK:      // %bb.0:
; CHECK-NEXT:  hint #34
; CHECK-NEXT:  ret
; CHECK-NOT:  .section __patchable_function_entries
  ret void
}

;; -fpatchable-function-entry=1 -mbranch-protection=bti
define void @f1() "patchable-function-entry"="1" "branch-target-enforcement" {
; CHECK-LABEL: f1:
; CHECK-NEXT: .Lfunc_begin1:
; CHECK:       hint #34
; CHECK-NEXT:  nop
; CHECK-NEXT:  ret
; CHECK:      .section __patchable_function_entries,"awo",@progbits,f1,unique,0
; CHECK-NEXT: .p2align 3
; CHECK-NEXT: .xword .Lfunc_begin1
  ret void
}

;; -fpatchable-function-entry=2,1 -mbranch-protection=bti
define void @f2_1() "patchable-function-entry"="1" "patchable-function-prefix"="1" "branch-target-enforcement" {
; CHECK-LABEL: .type f2_1,@function
; CHECK-NEXT: .Ltmp0:
; CHECK-NEXT:  nop
; CHECK-NEXT: f2_1:
; CHECK-NEXT: .Lfunc_begin2:
; CHECK:      // %bb.0:
; CHECK-NEXT:  hint #34
; CHECK-NEXT:  nop
; CHECK-NEXT:  ret
; CHECK:      .Lfunc_end2:
; CHECK-NEXT: .size f2_1, .Lfunc_end2-f2_1
; CHECK:      .section __patchable_function_entries,"awo",@progbits,f1,unique,0
; CHECK-NEXT: .p2align 3
; CHECK-NEXT: .xword .Ltmp0
  ret void
}
