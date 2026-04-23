; REQUIRES: asserts
; RUN: opt -S < %s -p loop-vectorize -debug-only=loop-vectorize --disable-output \
; RUN: -tail-folding-policy=fold-epilogue-tail 2>&1 | FileCheck %s

define void @test_epilogue_tf(ptr %A, i64 %n) {
; CHECK: LV: Checking a loop in 'test_epilogue_tf'
; CHECK: LV: epilogue tail-folding is not supported yet
; CHECK: The tail-folding policy fold-epilogue-tail is not supported yet, fall back to an epilogue.
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %A, i64 %iv
  store i8 1, ptr %arrayidx, align 1
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp ne i64 %iv.next, %n
  br i1 %exitcond, label %for.body, label %exit

exit:
  ret void
}
