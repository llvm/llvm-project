; RUN: llc -mtriple=x86_64-pc-windows-msvc -code-model=large < %s | FileCheck %s

;; On a COFF/MSVC target built with the large code model and split stacks, the
;; prologue calls __morestack indirectly through the __morestack_addr data slot,
;; whose definition is emitted at the end of the file by
;; X86AsmPrinter::emitEndOfAsmFile. When the program also uses floating point,
;; the same routine emits the _fltused marker. emitEndOfAsmFile used to `return`
;; right after emitting _fltused, which skipped the trailing __morestack_addr
;; definition -- leaving the indirect call referencing an undefined symbol.
;;
;; Both the _fltused marker and the __morestack_addr definition must be emitted.

; CHECK: callq *__morestack_addr(%rip)
; CHECK: .globl{{.*}}_fltused
; CHECK: __morestack_addr:
; CHECK-NEXT: .quad{{.*}}__morestack

target triple = "x86_64-pc-windows-msvc"

declare void @use(ptr)

define double @f(double %a, double %b) #0 {
  %buf = alloca [4096 x i8]
  call void @use(ptr %buf)
  %r = fadd double %a, %b
  ret double %r
}

attributes #0 = { "split-stack" }
