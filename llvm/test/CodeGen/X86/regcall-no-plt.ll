; RUN: llc -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-freebsd -relocation-model=pic < %s | FileCheck %s

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; According to x86-64 psABI, xmm0-xmm7 can be used to pass function parameters.  
;; However regcall calling convention uses also xmm8-xmm15 to pass function  
;; parameters which violates x86-64 psABI. 
;; Detail info about it can be found at:
;; https://sourceware.org/bugzilla/show_bug.cgi?id=21265
;;
;; We encounter the violation symptom when using PIC with lazy binding 
;; optimization.
;; In that case the PLT mechanism as described in x86_64 psABI will
;; not preserve xmm8-xmm15 registers and will lead to miscompilation.
;;
;; The agreed solution is to disable PLT for regcall calling convention for 
;; SystemV using ELF format.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

declare void @lazy()
declare x86_regcallcc void @regcall_not_lazy()

; CHECK-LABEL: foo:
; CHECK:  callq lazy@PLT
; CHECK:  callq *regcall_not_lazy@GOTPCREL(%rip)
define void @foo() nounwind {
  call void @lazy()
  call void @regcall_not_lazy()
  ret void
}

; CHECK-LABEL: tail_call_regcall:
; CHECK:   jmpq *regcall_not_lazy@GOTPCREL(%rip)
define void @tail_call_regcall() nounwind {
  tail call void @regcall_not_lazy()
  ret void
}

; CHECK-LABEL: tail_call_regular:
; CHECK:   jmp lazy
define void @tail_call_regular() nounwind {
  tail call void @lazy()
  ret void
}
