; RUN: opt %loadPolly -polly-import-jscop \
; RUN:   -polly-codegen -polly-invariant-load-hoisting -S \
; RUN:   2>&1 < %s | FileCheck %s

; Setting new access functions where the base pointer of the array that is newly
; accessed is only loaded within the scop itself caused incorrect code to be
; generated when invariant load hoisting is disabled. This test case checks
; that in case invariant load hoisting is enabled, we generate correct code.

; CHECK: %polly.access.polly.access.X.load = getelementptr float, ptr %polly.access.X.load, i64 %polly.indvar

define void @invariant_base_ptr(ptr noalias %Array, ptr noalias %X,
                                ptr noalias %C) {

start:
  br label %loop

loop:
  %indvar = phi i64 [0, %start], [%indvar.next, %latch]
  %indvar.next = add i64 %indvar, 1
  %cmp = icmp slt i64 %indvar, 1024
  br i1 %cmp, label %body, label %exit

body:
  %gep= getelementptr float, ptr %Array, i64 %indvar
  store float 42.0, ptr %gep
  br label %body2

body2:
  %Base = load ptr, ptr %X
  %gep2 = getelementptr float, ptr %Base, i64 %indvar
  %val2 = load float, ptr %gep2
  store float %val2, ptr %C
  br label %latch

latch:
  br label %loop

exit:
  ret void
}
