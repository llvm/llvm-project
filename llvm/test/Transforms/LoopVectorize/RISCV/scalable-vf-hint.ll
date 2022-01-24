; RUN: opt -mtriple=riscv64 -mattr=+m,+v -loop-vectorize \
; RUN:   -riscv-v-vector-bits-max=512 -S -scalable-vectorization=on < %s 2>&1 \
; RUN:   | FileCheck %s

; void test(int *a, int *b, int N) {
;   #pragma clang loop vectorize(enable) vectorize_width(2, scalable)
;   for (int i=0; i<N; ++i) {
;     a[i + 64] = a[i] + b[i];
;   }
; }
;
; CHECK: <vscale x 2 x i32>
define void @test(i32* %a, i32* %b) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %2 = add nuw nsw i64 %iv, 64
  %arrayidx5 = getelementptr inbounds i32, i32* %a, i64 %2
  store i32 %add, i32* %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !6

exit:
  ret void
}

!6 = !{!6, !7, !8}
!7 = !{!"llvm.loop.vectorize.width", i32 2}
!8 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
