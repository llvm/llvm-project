; RUN: llc -mtriple=powerpc-unknown-linux-gnu %s -o - | FileCheck %s --check-prefixes=CHECK,PPC32
; RUN: llc -mtriple=powerpc64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefixes=CHECK,PPC64

define void @f0() {
; CHECK-LABEL: f0:
; CHECK-NOT:   nop
; CHECK:       # %bb.0:
; CHECK-NEXT:    blr
; CHECK-NOT:   .section    __patchable_function_entries
  ret void
}

define void @f1() "patchable-function-entry"="0" {
; CHECK-LABEL: f1:
; CHECK-NOT:   nop
; CHECK:       # %bb.0:
; CHECK-NEXT:    blr
; CHECK-NOT:   .section    __patchable_function_entries
  ret void
}

define void @f2() "patchable-function-entry"="1" {
; CHECK-LABEL: f2:
; CHECK-LABEL-NEXT:  .Lfunc_begin2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    nop
; CHECK-NEXT:    blr
; CHECK:       .section    __patchable_function_entries
; PPC32:       .p2align    2, 0x0
; PPC64:       .p2align    3, 0x0
; PPC32-NEXT:  .long   .Lfunc_begin2
; PPC64-NEXT:  .quad   .Lfunc_begin2
  ret void
}

define void @f3() "patchable-function-entry"="1" "patchable-function-prefix"="2" {
; CHECK-LABEL: .Ltmp0:
; CHECK-COUNT-2: nop
; CHECK-LABEL: f3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    nop
; CHECK-NEXT:    blr
; CHECK:       .section    __patchable_function_entries
; PPC32:       .p2align    2, 0x0
; PPC64:       .p2align    3, 0x0
; PPC32-NEXT:  .long   .Ltmp0
; PPC64-NEXT:  .quad   .Ltmp0
  ret void
}
