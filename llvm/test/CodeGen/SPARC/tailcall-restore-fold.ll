; RUN: llc -mtriple=sparc64 < %s | FileCheck %s

;; DelaySlotFiller folds `add a, b, %iN; restore %g0, %g0, %g0` into
;; `restore a, b, %oN`. When %iN is the target of an indirect tail call the
;; fold must not happen: the jump would read only the ADD's first operand and
;; call base instead of base + offset, with the sum computed into an out
;; register nothing reads.

declare void @clobber()

define void @tailcall_computed_target(ptr %base, i64 %off) nounwind {
; CHECK-LABEL: tailcall_computed_target:
; CHECK:         add %i0, %i1, %[[T:i[0-7]]]
; CHECK-NEXT:    jmp %[[T]]
; CHECK-NEXT:    restore{{$}}
entry:
  call void @clobber()
  %tp = getelementptr i8, ptr %base, i64 %off
  tail call void %tp()
  ret void
}
