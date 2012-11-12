; RUN: llc < %s -mtriple=arm-apple-darwin | FileCheck %s -check-prefix=INSTR
; RUN: llc < %s -mtriple=arm-apple-darwin -trap-func=_trap | FileCheck %s -check-prefix=FUNC
; rdar://7961298
; rdar://9249183

define void @t() nounwind {
entry:
; INSTR: t:
; INSTR: trap

; FUNC: t:
; FUNC: bl __trap
  call void @llvm.trap()
  unreachable
}

define void @t2() nounwind {
entry:
; INSTR: t2:
; INSTR: trap

; FUNC: t2:
; FUNC: bl __trap
  call void @llvm.debugtrap()
  unreachable
}

declare void @llvm.trap() nounwind
declare void @llvm.debugtrap() nounwind
