; RUN: opt %loadPolly -polly-acc-dump-code -polly-invariant-load-hoisting -polly-print-scops -disable-output < %s | FileCheck %s -check-prefix=SCOP

; RUN: opt %loadPolly -S -polly-use-llvm-names -polly-codegen-ppcg \
; RUN: -polly-acc-dump-code -polly-stmt-granularity=bb \
; RUN: -polly-invariant-load-hoisting < %s | FileCheck %s -check-prefix=CODE

; RUN: opt %loadPolly -S -polly-use-llvm-names -polly-codegen-ppcg \
; RUN: -polly-invariant-load-hoisting -polly-stmt-granularity=bb < %s \
; RUN: | FileCheck %s -check-prefix=HOST-IR

; REQUIRES: pollyacc

; SCOP:      Invariant Accesses: {
; SCOP-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; SCOP-NEXT:             { Stmt_loop_a[i0] -> MemRef_p[0] };
; SCOP-NEXT:         Execution Context: {  :  }
; SCOP-NEXT: }

; CODE: # kernel0
; CODE-NEXT: {
; CODE-NEXT:   if (32 * b0 + t0 <= 1025) {
; CODE-NEXT:     Stmt_loop(32 * b0 + t0);
; CODE-NEXT:     write(0);
; CODE-NEXT:   }
; CODE-NEXT:   sync0();
; CODE-NEXT: }

; Check that we generate a correct "always false" branch.
; HOST-IR:  br i1 false, label %polly.start, label %loop.pre_entry_bb

; This test case checks that we generate correct code if PPCGCodeGeneration
; decides a build is unsuccessful with invariant load hoisting enabled.
;
; There is a conditional branch which switches between the original code and
; the new code. We try to set this conditional branch to branch on false.
; However, invariant load hoisting changes the structure of the scop, so we
; need to change the way we *locate* this instruction.

target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.12.0"

define void @foo(ptr %A, ptr %p) {
entry:
  br label %loop

loop:
  %indvar = phi i64 [0, %entry], [%indvar.next, %loop]
  %indvar.next = add i64 %indvar, 1
  %invariant = load float, ptr %p
  %ptr = getelementptr float, ptr %A, i64 %indvar
  store float 42.0, ptr %ptr
  %cmp = icmp sle i64 %indvar, 1024
  br i1 %cmp, label %loop, label %loop2

loop2:
  %indvar2 = phi i64 [0, %loop], [%indvar2.next, %loop2]
  %indvar2f = phi float [%invariant, %loop], [%indvar2f, %loop2]
  %indvar2.next = add i64 %indvar2, 1
  store float %indvar2f, ptr %A
  %cmp2 = icmp sle i64 %indvar2, 1024
  br i1 %cmp2, label %loop2, label %end

end:
  ret void
}
