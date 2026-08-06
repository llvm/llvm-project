; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize --disable-output \
; RUN: -tail-folding-policy=prefer-fold-tail -vectorizer-maximize-bandwidth \
; RUN: -mtriple=riscv64 -mattr=+v -S < %s 2>&1 | FileCheck %s

; CHECK: Cost of 0 for VF vscale x 4: WIDEN-REDUCTION-PHI ir<%rdx> = phi
; CHECK: Cost of 0 for VF vscale x 4: WIDEN-INTRINSIC vp<%{{.+}}> = call llvm.vp.merge(ir<true>, ir<%add>, ir<%rdx>, vp<%{{.+}}>)

; CHECK: Cost of 0 for VF vscale x 8: WIDEN-REDUCTION-PHI ir<%rdx> = phi
; CHECK: Cost of 0 for VF vscale x 8: WIDEN-INTRINSIC vp<%{{.+}}> = call llvm.vp.merge(ir<true>, ir<%add>, ir<%rdx>, vp<%{{.+}}>)

; Type needs split, won't be folded:

; CHECK: Cost of 0 for VF vscale x 16: WIDEN-REDUCTION-PHI ir<%rdx> = phi
; CHECK: Cost of 16 for VF vscale x 16: WIDEN-INTRINSIC vp<%{{.+}}> = call llvm.vp.merge(ir<true>, ir<%add>, ir<%rdx>, vp<%{{.+}}>)

define i64 @add(ptr %a, i64 %n, i64 %start) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %rdx = phi i64 [ %start, %entry ], [ %add, %loop ]
  %arrayidx = getelementptr inbounds i8, ptr %a, i64 %iv
  %0 = load i8, ptr %arrayidx
  %zext = zext i8 %0 to i64
  %add = add nsw i64 %zext, %rdx
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret i64 %add
}
