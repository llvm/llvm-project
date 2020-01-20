; RUN: llc -mtriple=aarch64 %s -o - | FileCheck --check-prefixes=CHECK %s

define i32 @f0() "patchable-function-entry"="0" "branch-target-enforcement" {
; CHECK-LABEL: f0:
; CHECK-NEXT: .Lfunc_begin0:
; CHECK:       hint #34
; CHECK-NEXT:  mov w0, wzr
; CHECK:       .section __patchable_function_entries,"awo",@progbits,f0,unique,0
; CHECK-NEXT:  .p2align 3
; CHECK-NEXT:  .xword .Lfunc_begin0
  ret i32 0
}

define i32 @f1() "patchable-function-entry"="1" "branch-target-enforcement" {
; CHECK-LABEL: f1:
; CHECK-NEXT: .Lfunc_begin1:
; CHECK:       hint #34
; CHECK-NEXT:  nop
; CHECK-NEXT:  mov w0, wzr
; CHECK:       .section __patchable_function_entries,"awo",@progbits,f0,unique,0
; CHECK-NEXT:  .p2align 3
; CHECK-NEXT:  .xword .Lfunc_begin1
  ret i32 0
}
