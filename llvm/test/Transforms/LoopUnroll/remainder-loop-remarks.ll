; RUN: opt < %s -S -passes=loop-unroll -pass-remarks=loop-unroll -pass-remarks-missed=loop-unroll -unroll-count=4 2>&1 | FileCheck %s

; Verify that the unroller uses vectorizer metadata to produce more informative
; remarks distinguishing vectorized loops from their scalar remainders.

; CHECK: remark: {{.*}}: unrolled scalar remainder loop after vectorization by a factor of 4
define i32 @remainder_loop(i32 %n) {
entry:
  br label %for.body

for.body:
  %s = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %add = add nsw i32 %i, %s
  %inc = add nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret i32 %add
}

; CHECK: remark: {{.*}}: unrolled vectorized loop by a factor of 4
define i32 @vectorized_loop() {
entry:
  br label %for.body

for.body:
  %s = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %add = add nsw i32 %i, %s
  %inc = add nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, 64
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !3

for.end:
  ret i32 %add
}

; CHECK: remark: {{.*}}: unrolled loop by a factor of 4
define i32 @plain_loop() {
entry:
  br label %for.body

for.body:
  %s = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %add = add nsw i32 %i, %s
  %inc = add nsw i32 %i, 1
  %exitcond = icmp eq i32 %inc, 64
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 %add
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.isvectorized", i32 1}
!2 = !{!"llvm.loop.vectorize.scalar_remainder", i32 1}
!3 = distinct !{!3, !1, !4}
!4 = !{!"llvm.loop.vectorize.vector_body", i32 1}
