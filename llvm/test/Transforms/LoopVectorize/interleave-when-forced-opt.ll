; REQUIRES: asserts
; RUN: opt -passes='loop-vectorize<interleave-forced-only>' -debug-only=loop-vectorize --disable-output -S %s 2>&1 | FileCheck --check-prefix=CHECK-INTERLEAVE-FORCED %s
; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize --disable-output -S %s 2>&1 | FileCheck --check-prefix=CHECK-INTERLEAVE-NOT-FORCED %s


; CHECK-INTERLEAVE-FORCED-LABEL: LV: Checking a loop in 'test_interleave_when_forced_opt'
; CHECK-INTERLEAVE-FORCED:  LV: Interleaving disabled by the pass manager

; CHECK-INTERLEAVE-NOT-FORCED-LABEL: LV: Checking a loop in 'test_interleave_when_forced_opt'
; CHECK-INTERLEAVE-NOT-FORCED-NOT:  LV: Interleaving disabled by the pass manager

define void @test_interleave_when_forced_opt(ptr %dst, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add i64 %iv, 1
  %gep = getelementptr i8, ptr %dst, i64 %iv
  store i8 0, ptr %gep, align 1
  %ec = icmp ult i64 %iv.next, %n
  br i1 %ec, label %loop, label %exit

exit:
  ret void
}
