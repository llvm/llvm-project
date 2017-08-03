; RUN: llc < %s -relocation-model=pic -mtriple=x86_64-apple-darwin10 | FileCheck %s -check-prefix=PIC64
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -relocation-model=static | FileCheck %s -check-prefix=STATIC64

; Use %rip-relative addressing even in static mode on x86-64, because
; it has a smaller encoding.

@a = internal global double 3.4
define double @foo() nounwind {
  %a = load double, double* @a
  ret double %a
  
; PIC64:    movsd	_a(%rip), %xmm0
; STATIC64: movsd	a(%rip), %xmm0
}
