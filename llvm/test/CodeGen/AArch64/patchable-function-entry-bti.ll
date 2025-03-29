; RUN: llc -mtriple=aarch64 -aarch64-min-jump-table-entries=4 %s -o - | FileCheck %s

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
;; For M=0, place the label .Lpatch0 after the initial BTI.
define void @f1() "patchable-function-entry"="1" "branch-target-enforcement" {
; CHECK-LABEL: f1:
; CHECK-NEXT: .Lfunc_begin1:
; CHECK-NEXT: .cfi_startproc
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  hint #34
; CHECK-NEXT: .Lpatch0:
; CHECK-NEXT:  nop
; CHECK-NEXT:  ret
; CHECK:      .section __patchable_function_entries,"awo",@progbits,f1{{$}}
; CHECK-NEXT: .p2align 3
; CHECK-NEXT: .xword .Lpatch0
  ret void
}

;; -fpatchable-function-entry=2,1 -mbranch-protection=bti
define void @f2_1() "patchable-function-entry"="1" "patchable-function-prefix"="1" "branch-target-enforcement" {
; CHECK-LABEL: .type f2_1,@function
; CHECK-NEXT: .Ltmp0:
; CHECK-NEXT:  nop
; CHECK-NEXT: f2_1:
; CHECK-NEXT: .Lfunc_begin2:
; CHECK-NEXT: .cfi_startproc
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT:  hint #34
; CHECK-NEXT:  nop
; CHECK-NEXT:  ret
; CHECK:      .Lfunc_end2:
; CHECK-NEXT: .size f2_1, .Lfunc_end2-f2_1
; CHECK:      .section __patchable_function_entries,"awo",@progbits,f2_1{{$}}
; CHECK-NEXT: .p2align 3
; CHECK-NEXT: .xword .Ltmp0
  ret void
}

;; -fpatchable-function-entry=1 -mbranch-protection=bti
;; We add BTI c even when the function has internal linkage
define internal void @f1i(i64 %v) "patchable-function-entry"="1" "branch-target-enforcement" {
; CHECK-LABEL: f1i:
; CHECK-NEXT: .Lfunc_begin3:
; CHECK:      // %bb.0:
; CHECK-NEXT:  hint #34
; CHECK-NEXT:  .Lpatch1:
; CHECK-NEXT:  nop
;; Other basic blocks have BTI, but they don't affect our decision to not create .Lpatch0
; CHECK:      .LBB{{.+}} // %sw.bb1
; CHECK-NEXT:  hint #36
; CHECK:      .section __patchable_function_entries,"awo",@progbits,f1i{{$}}
; CHECK-NEXT: .p2align 3
; CHECK-NEXT: .xword .Lpatch1
entry:
  switch i64 %v, label %sw.bb0 [
    i64 1, label %sw.bb1
    i64 2, label %sw.bb2
    i64 3, label %sw.bb3
    i64 4, label %sw.bb4
  ]
sw.bb0:
  call void asm sideeffect "nop", ""()
  ret void
sw.bb1:
  call void asm sideeffect "", ""()
  ret void
sw.bb2:
  call void asm sideeffect "", ""()
  ret void
sw.bb3:
  call void asm sideeffect "", ""()
  ret void
sw.bb4:
  call void asm sideeffect "", ""()
  ret void
}

;; Test the interaction with -fsanitize=function.
; CHECK:      .type sanitize_function,@function
; CHECK-NEXT: .Ltmp{{.*}}:
; CHECK-NEXT:   nop
; CHECK-NEXT:   .word   3238382334  // 0xc105cafe
; CHECK-NEXT:   .word   42
; CHECK-NEXT: sanitize_function:
; CHECK-NEXT: .Lfunc_begin{{.*}}:
; CHECK-NEXT:   .cfi_startproc
; CHECK-NEXT:   // %bb.0:
; CHECK-NEXT:   hint #34
; CHECK-NEXT:   nop
; CHECK-NEXT:   ret
define void @sanitize_function(ptr noundef %x) "patchable-function-prefix"="1" "patchable-function-entry"="1" "branch-target-enforcement" !func_sanitize !0 {
  ret void
}

!0 = !{i32 3238382334, i32 42}
