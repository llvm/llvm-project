; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -S < %s 2>&1 | FileCheck %s

; CHECK: Cost of 2 for VF vscale x 4: WIDEN-INTRINSIC vp<%{{.+}}> = call llvm.vp.merge(ir<true>, ir<%add>, ir<%rdx>, vp<%{{.+}}>)
; CHECK: LV: Found an estimated cost of 2 for VF vscale x 4 For instruction:   %rdx = phi i32 [ %start, %entry ], [ %add, %loop ]

define i32 @add(ptr %a, i64 %n, i32 %start) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %rdx = phi i32 [ %start, %entry ], [ %add, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %rdx
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret i32 %add
}
