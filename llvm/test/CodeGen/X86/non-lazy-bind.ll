; RUN: llc -mtriple=x86_64-apple-darwin < %s | FileCheck %s

declare void @lazy() nonlazybind
declare void @not()

; CHECK: foo:
; CHECK:  callq _not
; CHECK:  callq *_lazy@GOTPCREL(%rip)
define void @foo() nounwind {
  call void @not()
  call void @lazy()
  ret void
}

; CHECK: tail_call_regular:
; CHECK:   jmp _not
define void @tail_call_regular() nounwind {
  tail call void @not()
  ret void
}

; CHECK: tail_call_eager:
; CHECK:   jmpq *_lazy@GOTPCREL(%rip)
define void @tail_call_eager() nounwind {
  tail call void @lazy()
  ret void
}
