; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-dir=%S \
; RUN:   -polly-codegen -analyze 2 >&1 < %s | FileCheck %s

; XFAIL: *

; Setting new access functions where the base pointer of the array that is newly
; accessed is only loaded within the scop itself caused incorrect code to be
; generated when invariant load hoisting is disabled. Since r282893 we assert
; in such situations. This test case was added to demonstrate what needs to be
; resolved to support such access functions.

; CHECK: %polly.access.polly.access.X.load = getelementptr float, float* %polly.access.X.load, i64 %polly.indvar

define void @invariant_base_ptr(float* noalias %Array, float** noalias %X,
                                float* noalias %C) {

start:
  br label %loop

loop:
  %indvar = phi i64 [0, %start], [%indvar.next, %latch]
  %indvar.next = add i64 %indvar, 1
  %cmp = icmp slt i64 %indvar, 1024
  br i1 %cmp, label %body, label %exit

body:
  %gep= getelementptr float, float* %Array, i64 %indvar
  store float 42.0, float* %gep
  br label %body2

body2:
  %Base = load float*, float** %X
  %gep2 = getelementptr float, float* %Base, i64 %indvar
  %val2 = load float, float* %gep2
  store float %val2, float* %C
  br label %latch

latch:
  br label %loop

exit:
  ret void
}
