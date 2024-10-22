; RUN: llc -mtriple=i386 %s -o - | FileCheck --check-prefixes=CHECK,X86 %s
; RUN: llc -mtriple=x86_64 %s -o - | FileCheck --check-prefixes=CHECK,X64 %s
; RUN: llc -mtriple=x86_64 -function-sections %s -o - | FileCheck --check-prefixes=CHECK,X64 %s

define void @f0() "patchable-function-entry"="0" {
; CHECK-LABEL: f0:
; CHECK-NEXT: .Lfunc_begin0:
; CHECK-NOT:   nop
; CHECK:       ret
; CHECK-NOT:   .section __patchable_function_entries
  ret void
}

define void @f1() "patchable-function-entry"="1" {
; CHECK-LABEL: f1:
; CHECK-NEXT: .Lfunc_begin1:
; CHECK:       nop
; CHECK-NEXT:  ret
; CHECK:       .section __patchable_function_entries,"awo",@progbits,f1{{$}}
; X86:          .p2align 2
; X86-NEXT:     .long .Lfunc_begin1
; X64:          .p2align 3
; X64-NEXT:     .quad .Lfunc_begin1
  ret void
}

;; Without -function-sections, f2 is in the same text section as f1.
;; They share the __patchable_function_entries section.
;; With -function-sections, f1 and f2 are in different text sections.
;; Use separate __patchable_function_entries.
define void @f2() "patchable-function-entry"="2" {
; CHECK-LABEL: f2:
; CHECK-NEXT: .Lfunc_begin2:
; X86:          xchgw %ax, %ax
; X64:          xchgw %ax, %ax
; CHECK-NEXT:  ret
; CHECK:       .section __patchable_function_entries,"awo",@progbits,f2{{$}}
; X86:          .p2align 2
; X86-NEXT:     .long .Lfunc_begin2
; X64:          .p2align 3
; X64-NEXT:     .quad .Lfunc_begin2
  ret void
}

$f3 = comdat any
define void @f3() "patchable-function-entry"="3" comdat {
; CHECK-LABEL: f3:
; CHECK-NEXT: .Lfunc_begin3:
; X86:          xchgw %ax, %ax
; X86-NEXT:     nop
; X64:          nopl (%rax)
; CHECK:       ret
; CHECK:       .section __patchable_function_entries,"awoG",@progbits,f3,f3,comdat{{$}}
; X86:          .p2align 2
; X86-NEXT:     .long .Lfunc_begin3
; X64:          .p2align 3
; X64-NEXT:     .quad .Lfunc_begin3
  ret void
}

$f5 = comdat any
define void @f5() "patchable-function-entry"="5" comdat {
; CHECK-LABEL: f5:
; CHECK-NEXT: .Lfunc_begin4:
; X86-COUNT-2:  xchgw %ax, %ax
; X86-NEXT:     nop
; X64:          nopl 8(%rax,%rax)
; CHECK-NEXT:  ret
; CHECK:       .section __patchable_function_entries,"awoG",@progbits,f5,f5,comdat{{$}}
; X86:          .p2align 2
; X86-NEXT:     .long .Lfunc_begin4
; X64:          .p2align 3
; X64-NEXT:     .quad .Lfunc_begin4
  ret void
}

;; -fpatchable-function-entry=3,2
;; "patchable-function-prefix" emits data before the function entry label.
;; We emit 1-byte NOPs before the function entry, so that with a partial patch,
;; the remaining instructions do not need to be modified.
define void @f3_2() "patchable-function-entry"="1" "patchable-function-prefix"="2" {
; CHECK-LABEL: .type f3_2,@function
; CHECK-NEXT: .Ltmp0:
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT: f3_2: # @f3_2
; CHECK:      # %bb.0:
; CHECK-NEXT:  nop
; CHECK-NEXT:  ret
;; .size does not include the prefix.
; CHECK:      .Lfunc_end5:
; CHECK-NEXT: .size f3_2, .Lfunc_end5-f3_2
; CHECK:      .section __patchable_function_entries,"awo",@progbits,f3_2{{$}}
; X86:         .p2align 2
; X86-NEXT:    .long .Ltmp0
; X64:         .p2align 3
; X64-NEXT:    .quad .Ltmp0
  %frame = alloca i8, i32 16
  ret void
}
