;; Test the function attribute "patchable-function-entry".
;; Adapted from the RISCV test case.
; RUN: llc --mtriple=loongarch32 -mattr=+d < %s | FileCheck %s --check-prefixes=CHECK,LA32
; RUN: llc --mtriple=loongarch64 -mattr=+d < %s | FileCheck %s --check-prefixes=CHECK,LA64

define void @f0() "patchable-function-entry"="0" {
; CHECK-LABEL: f0:
; CHECK-NEXT:  .Lfunc_begin0:
; CHECK-NOT:     nop
; CHECK:         ret
; CHECK-NOT:   .section __patchable_function_entries
  ret void
}

define void @f1() "patchable-function-entry"="1" {
; CHECK-LABEL: f1:
; CHECK-NEXT: .Lfunc_begin1:
; CHECK:         nop
; CHECK-NEXT:    ret
; CHECK:       .section __patchable_function_entries,"awo",@progbits,f1{{$}}
; LA32:        .p2align 2
; LA32-NEXT:   .word .Lfunc_begin1
; LA64:        .p2align 3
; LA64-NEXT:   .dword .Lfunc_begin1
  ret void
}

$f5 = comdat any
define void @f5() "patchable-function-entry"="5" comdat {
; CHECK-LABEL:   f5:
; CHECK-NEXT:    .Lfunc_begin2:
; CHECK-COUNT-5:   nop
; CHECK-NEXT:      ret
; CHECK:         .section __patchable_function_entries,"awoG",@progbits,f5,f5,comdat{{$}}
; LA32:          .p2align 2
; LA32-NEXT:     .word .Lfunc_begin2
; LA64:          .p2align 3
; LA64-NEXT:     .dword .Lfunc_begin2
  ret void
}

;; -fpatchable-function-entry=3,2
;; "patchable-function-prefix" emits data before the function entry label.
define void @f3_2() "patchable-function-entry"="1" "patchable-function-prefix"="2" {
; CHECK-LABEL:   .type f3_2,@function
; CHECK-NEXT:    .Ltmp0:
; CHECK-COUNT-2:   nop
; CHECK-NEXT:    f3_2:  # @f3_2
; CHECK:         # %bb.0:
; CHECK-NEXT:      nop
; LA32-NEXT:       addi.w $sp, $sp, -16
; LA64-NEXT:       addi.d $sp, $sp, -16
;; .size does not include the prefix.
; CHECK:      .Lfunc_end3:
; CHECK-NEXT: .size f3_2, .Lfunc_end3-f3_2
; CHECK:      .section __patchable_function_entries,"awo",@progbits,f3_2{{$}}
; LA32:       .p2align 2
; LA32-NEXT:  .word .Ltmp0
; LA64:       .p2align 3
; LA64-NEXT:  .dword .Ltmp0
  %frame = alloca i8, i32 16
  ret void
}
