; RUN: llc -mtriple=powerpc %s -o - | FileCheck %s --check-prefixes=CHECK,PPC32
; RUN: llc -mtriple=powerpc64 %s -o - | FileCheck %s --check-prefixes=CHECK,PPC64

@a = global i32 0, align 4

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

define i32 @f3() "patchable-function-entry"="1" "patchable-function-prefix"="2" {
; CHECK-LABEL: .Ltmp0:
; CHECK-COUNT-2: nop
; CHECK-LABEL: f3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    nop
; PPC32:         lis 3, a@ha
; PPC32-NEXT:    lwz 3, a@l(3)
; PPC64:         addis 3, 2, .LC0@toc@ha
; PPC64-NEXT:    ld 3, .LC0@toc@l(3)
; PPC64-NEXT:    lwz 3, 0(3)
; CHECK:         blr
; CHECK:       .section    __patchable_function_entries
; PPC32:       .p2align    2, 0x0
; PPC64:       .p2align    3, 0x0
; PPC32-NEXT:  .long   .Ltmp0
; PPC64-NEXT:  .quad   .Ltmp0
entry:
  %0 = load i32, ptr @a, align 4
  ret i32 %0
}
