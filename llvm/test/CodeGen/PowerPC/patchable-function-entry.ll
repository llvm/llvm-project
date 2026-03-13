; RUN: llc -mtriple=powerpc %s -o - | FileCheck %s --check-prefixes=CHECK,PPC32
; RUN: llc -mtriple=powerpc64 %s -o - | FileCheck %s --check-prefixes=CHECK,PPC64
; RUN: llc -mtriple=powerpc64le %s -o - | FileCheck %s --check-prefix=PPC64LE

@a = global i32 0, align 4

define void @f0() {
; CHECK-LABEL: f0:
; CHECK-NOT:   nop
; CHECK:       # %bb.0:
; CHECK-NEXT:    blr
; CHECK-NOT:   .section    __patchable_function_entries
;
; PPC64LE-LABEL: f0:
; PPC64LE-NOT:   nop
; PPC64LE:       # %bb.0:
; PPC64LE-NEXT:  blr
; PPC64LE-NOT:   .section    __patchable_function_entries
  ret void
}

define void @f1() "patchable-function-entry"="0" {
; CHECK-LABEL: f1:
; CHECK-NOT:   nop
; CHECK:       # %bb.0:
; CHECK-NEXT:    blr
; CHECK-NOT:   .section    __patchable_function_entries
;
; PPC64LE-LABEL: f1:
; PPC64LE:        # %bb.0:
; PPC64LE-NEXT:   .Ltmp0:
; PPC64LE-NEXT:   b .Ltmp1
; PPC64LE-NEXT:   nop
; PPC64LE-NEXT:   std 0, -8(1)
; PPC64LE-NEXT:   mflr 0
; PPC64LE-NEXT:   bl __xray_FunctionEntry
; PPC64LE-NEXT:   nop
; PPC64LE-NEXT:   mtlr 0
; PPC64LE-NEXT:   .Ltmp1:
; PPC64LE-NEXT:   blr
; PPC64LE-NOT:    .section    __patchable_function_entries
; PPC64LE:     .section        xray_instr_map
; PPC64LE:     .section        xray_fn_idx
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
;
; PPC64LE-LABEL: f2:
; PPC64LE-LABEL-NEXT:  .Lfunc_begin2:
; PPC64LE:         # %bb.0:
; PPC64LE-NEXT:    nop
; PPC64LE-NEXT:    blr
; PPC64LE:        .section    __patchable_function_entries
; PPC64LE:        .p2align    3, 0x0
; PPC64LE-NEXT:   .quad   .Lfunc_begin2
; PPC64LE-NOT:    .section        xray_instr_map
; PPC64LE-NOT:    .section        xray_fn_idx
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
;
; PC64LE-LABEL:    .Ltmp3:
; PC64LE-COUNT-2:  nop
; PC64LE-LABEL:    f3:
; PC64LE:          # %bb.0:
; PC64LE-NEXT:     nop
; PC64LE:          addis 3, 2, .LC0@toc@ha
; PC64LE-NEXT:     ld 3, .LC0@toc@l(3)
; PC64LE-NEXT:     lwz 3, 0(3)
; PC64LE:          blr
; PC64LE:         .section    __patchable_function_entries
; PPC64LE:        .p2align    3, 0x0
; PPC64LE-NEXT:   .quad   .Ltmp3
; PC64LE-NOT:     .section    xray_instr_map
; PC64LE-NOT:     .section    xray_fn_idx
entry:
  %0 = load i32, ptr @a, align 4
  ret i32 %0
}
