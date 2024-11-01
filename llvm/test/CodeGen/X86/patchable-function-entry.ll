; RUN: llc -mtriple=i386 %s -o - | FileCheck --check-prefixes=CHECK,32 %s
; RUN: llc -mtriple=x86_64 %s -o - | FileCheck --check-prefixes=CHECK,64 %s
; RUN: llc -mtriple=x86_64 -function-sections %s -o - | FileCheck --check-prefixes=CHECK,64 %s

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
; 32:          .p2align 2
; 32-NEXT:     .long .Lfunc_begin1
; 64:          .p2align 3
; 64-NEXT:     .quad .Lfunc_begin1
  ret void
}

;; Without -function-sections, f2 is in the same text section as f1.
;; They share the __patchable_function_entries section.
;; With -function-sections, f1 and f2 are in different text sections.
;; Use separate __patchable_function_entries.
define void @f2() "patchable-function-entry"="2" {
; CHECK-LABEL: f2:
; CHECK-NEXT: .Lfunc_begin2:
; 32:          xchgw %ax, %ax
; 64:          xchgw %ax, %ax
; CHECK-NEXT:  ret
; CHECK:       .section __patchable_function_entries,"awo",@progbits,f2{{$}}
; 32:          .p2align 2
; 32-NEXT:     .long .Lfunc_begin2
; 64:          .p2align 3
; 64-NEXT:     .quad .Lfunc_begin2
  ret void
}

$f3 = comdat any
define void @f3() "patchable-function-entry"="3" comdat {
; CHECK-LABEL: f3:
; CHECK-NEXT: .Lfunc_begin3:
; 32:          xchgw %ax, %ax
; 32-NEXT:     nop
; 64:          nopl (%rax)
; CHECK:       ret
; CHECK:       .section __patchable_function_entries,"aGwo",@progbits,f3,comdat,f3{{$}}
; 32:          .p2align 2
; 32-NEXT:     .long .Lfunc_begin3
; 64:          .p2align 3
; 64-NEXT:     .quad .Lfunc_begin3
  ret void
}

$f5 = comdat any
define void @f5() "patchable-function-entry"="5" comdat {
; CHECK-LABEL: f5:
; CHECK-NEXT: .Lfunc_begin4:
; 32-COUNT-2:  xchgw %ax, %ax
; 32-NEXT:     nop
; 64:          nopl 8(%rax,%rax)
; CHECK-NEXT:  ret
; CHECK:       .section __patchable_function_entries,"aGwo",@progbits,f5,comdat,f5{{$}}
; 32:          .p2align 2
; 32-NEXT:     .long .Lfunc_begin4
; 64:          .p2align 3
; 64-NEXT:     .quad .Lfunc_begin4
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
; 32:         .p2align 2
; 32-NEXT:    .long .Ltmp0
; 64:         .p2align 3
; 64-NEXT:    .quad .Ltmp0
  %frame = alloca i8, i32 16
  ret void
}
