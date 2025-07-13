; RUN: opt -passes=loop-vectorize -mcpu=grace -S %s -o /dev/null
; REQUIRES: aarch64-registered-target

; Reduced from GitHub issue #148389.
; The test just needs to run – if the pass crashes the test will fail.

define void @h(ptr %e, ptr %f, i8 %d) {
entry:
  br label %body
body:
  %i   = phi i16 [0, %entry], [%inc, %latch]
  %idx = sext i16 %i to i64
  %eelt = getelementptr [1 x i16], ptr %e, i64 %idx, i64 %idx
  %eval = load i16, ptr %eelt
  %felt = getelementptr [7 x i8],  ptr %f, i64 %idx, i64 %idx
  %fval = load i8,  ptr %felt
  %cmp  = icmp eq i8 %d, 0
  br i1 %cmp, label %update, label %latch
update:
  br label %latch
latch:
  %inc  = add nuw nsw i16 %i, 1
  %exit = icmp eq i16 %inc, 17
  br i1 %exit, label %exit.block, label %body
exit.block:
  ret void
}
