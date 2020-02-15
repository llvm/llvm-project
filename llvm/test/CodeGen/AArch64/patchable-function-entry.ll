; RUN: llc -mtriple=aarch64 %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64 -function-sections %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64 -no-integrated-as %s -o - | FileCheck --check-prefix=NOLINK %s

;; GNU as < 2.35 did not support section flag 'o'.
; NOLINK-NOT: "awo"

define i32 @f0() "patchable-function-entry"="0" {
; CHECK-LABEL: f0:
; CHECK-NEXT: .Lfunc_begin0:
; CHECK-NOT:   nop
; CHECK:       mov w0, wzr
; CHECK-NOT:   .section __patchable_function_entries
  ret i32 0
}

define i32 @f1() "patchable-function-entry"="1" {
; CHECK-LABEL: f1:
; CHECK-NEXT: .Lfunc_begin1:
; CHECK:       nop
; CHECK-NEXT:  mov w0, wzr
; CHECK:       .section __patchable_function_entries,"awo",@progbits,f1{{$}}
; CHECK-NEXT:  .p2align 3
; CHECK-NEXT:  .xword .Lfunc_begin1
  ret i32 0
}

;; Without -function-sections, f2 is in the same text section as f1.
;; They share the __patchable_function_entries section.
;; With -function-sections, f1 and f2 are in different text sections.
;; Use separate __patchable_function_entries.
define void @f2() "patchable-function-entry"="2" {
; CHECK-LABEL: f2:
; CHECK-NEXT: .Lfunc_begin2:
; CHECK-COUNT-2: nop
; CHECK-NEXT:  ret
; CHECK:       .section __patchable_function_entries,"awo",@progbits,f2{{$}}
; CHECK-NEXT:  .p2align 3
; CHECK-NEXT:  .xword .Lfunc_begin2
  ret void
}

$f3 = comdat any
define void @f3() "patchable-function-entry"="3" comdat {
; CHECK-LABEL: f3:
; CHECK-NEXT: .Lfunc_begin3:
; CHECK-COUNT-3: nop
; CHECK-NEXT:  ret
; CHECK:       .section __patchable_function_entries,"aGwo",@progbits,f3,comdat,f3{{$}}
; CHECK-NEXT:  .p2align 3
; CHECK-NEXT:  .xword .Lfunc_begin3
  ret void
}

$f5 = comdat any
define void @f5() "patchable-function-entry"="5" comdat {
; CHECK-LABEL: f5:
; CHECK-NEXT: .Lfunc_begin4:
; CHECK-COUNT-5: nop
; CHECK-NEXT:  sub sp, sp, #16
; CHECK:       .section __patchable_function_entries,"aGwo",@progbits,f5,comdat,f5{{$}}
; CHECK:       .p2align 3
; CHECK-NEXT:  .xword .Lfunc_begin4
  %frame = alloca i8, i32 16
  ret void
}

;; -fpatchable-function-entry=3,2
;; "patchable-function-prefix" emits data before the function entry label.
define void @f3_2() "patchable-function-entry"="1" "patchable-function-prefix"="2" {
; CHECK-LABEL: .type f3_2,@function
; CHECK-NEXT: .Ltmp1: // @f3_2
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT: f3_2:
; CHECK:      // %bb.0:
; CHECK-NEXT:  nop
; CHECK-NEXT:  ret
;; .size does not include the prefix.
; CHECK:      .Lfunc_end5:
; CHECK-NEXT: .size f3_2, .Lfunc_end5-f3_2
; CHECK:      .section __patchable_function_entries,"awo",@progbits,f3_2{{$}}
; CHECK:      .p2align 3
; CHECK-NEXT: .xword .Ltmp1
  ret void
}

;; When prefix data is used, arbitrarily place NOPs after prefix data.
define void @prefix() "patchable-function-entry"="0" "patchable-function-prefix"="1" prefix i32 1 {
; CHECK-LABEL: .type prefix,@function
; CHECK-NEXT: .word 1 // @prefix
; CHECK:      .Ltmp2:
; CHECK:       nop
; CHECK-NEXT: prefix:
;; Emit a __patchable_function_entries entry even if "patchable-function-entry" is 0.
; CHECK:      .section __patchable_function_entries,"awo",@progbits,prefix{{$}}
; CHECK:      .p2align 3
; CHECK-NEXT: .xword .Ltmp2
  ret void
}
