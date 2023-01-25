; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=simplify-conditionals-true --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=RESULT-TRUE %s < %t

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=simplify-conditionals-false --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=RESULT-FALSE %s < %t

; CHECK-INTERESTINGNESS-LABEL: @func(
; CHECK-INTERESTINGNESS: store i32 1,

; RESULT-TRUE: bb0:
; RESULT-TRUE: store i32 0, ptr null, align 4
; RESULT-TRUE-NEXT: store i32 1, ptr null, align 4
; RESULT-TRUE-NEXT: br label %bb2
; RESULT-TRUE-NOT: bb1


; RESULT-FALSE: bb0:
; RESULT-FALSE: store i32 0, ptr null, align 4
; RESULT-FALSE-NEXT: br label %bb2

; RESULT-FALSE: bb1: ; No predecessors!
; RESULT-FALSE-NEXT: store i32 1, ptr null, align 4
; RESULT-FALSE-NEXT: br label %bb3
define void @func(i1 %cond0, i1 %cond1) {
bb0:
  store i32 0, ptr null
  br i1 %cond0, label %bb1, label %bb2

bb1:
  store i32 1, ptr null
  br i1 %cond1, label %bb2, label %bb3

bb2:
  store i32 2, ptr null
  br label %bb3

bb3:
  ret void
}
