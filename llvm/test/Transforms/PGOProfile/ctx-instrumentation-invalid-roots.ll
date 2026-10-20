; RUN: split-file %s %t
; RUN: not opt -passes=ctx-instr-gen,ctx-instr-lower -profile-context-root=the_func -S %t/musttail.ll -o - 2>&1 | FileCheck %s
; RUN: not opt -passes=ctx-instr-gen,ctx-instr-lower -profile-context-root=the_func -S %t/unreachable.ll -o - 2>&1 | FileCheck %s
; RUN: not opt -passes=ctx-instr-gen,ctx-instr-lower -profile-context-root=the_func -S %t/noreturn.ll -o - 2>&1 | FileCheck %s

;--- musttail.ll
declare void @foo()

define void @the_func() {
  musttail call void @foo()
  ret void
}
;--- unreachable.ll
define void @the_func() {
  unreachable
}
;--- noreturn.ll
define void @the_func() noreturn {
  unreachable
}

; CHECK: error: [ctxprof] The function the_func was indicated as context root
