; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; A tail call that grows the stack-argument area makes the prologue reserve the
; extra space directly below the return address. The reserve is its own
; allocation as far as the unwinder is concerned, and it is probed like the main
; allocation.

declare tailcc void @g(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define tailcc void @reserve(i64 %a, i64 %b) {
; CHECK-LABEL: reserve:
; CHECK:         subq $48, %rsp
; CHECK-NEXT:    .seh_stackalloc 48
; CHECK-NEXT:    .seh_endprologue
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
}

define tailcc void @reserve_probed(i64 %a, i64 %b) "stack-probe-size"="32" {
; CHECK-LABEL: reserve_probed:
; CHECK:         movl $48, %eax
; CHECK-NEXT:    callq __chkstk
; CHECK-NEXT:    subq %rax, %rsp
; CHECK-NEXT:    .seh_stackalloc 48
; CHECK-NEXT:    .seh_endprologue
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
}

define tailcc void @reserve_no_probe(i64 %a, i64 %b) "stack-probe-size"="32" "no-stack-arg-probe" {
; CHECK-LABEL: reserve_no_probe:
; CHECK:         subq $48, %rsp
; CHECK-NEXT:    .seh_stackalloc 48
; CHECK-NEXT:    .seh_endprologue
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
}
