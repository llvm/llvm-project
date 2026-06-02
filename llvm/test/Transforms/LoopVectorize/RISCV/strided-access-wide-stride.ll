; RUN: not --crash opt < %s -passes=loop-vectorize -mtriple=riscv64 -mattr=+v,+zvl256b -S 2>&1 | FileCheck %s
; CHECK: isIntN

; Test for https://github.com/llvm/llvm-project/issues/199640
; convertToStridedAccesses uses the canonical IV type (i32) for the stride
; constant, but the pointer stride exceeds INT32_MAX on rv64 with a narrow IV.

define void @stride_exceeds_i32_max(ptr noalias readonly %src, ptr noalias %dst, i16 %start) {
entry:
  br label %loop

loop:
  %iv = phi i16 [ %start, %entry ], [ %iv.next, %loop ]
  %iv.ext = sext i16 %iv to i64
  %offset = mul nsw i64 %iv.ext, 3000000000
  %gep = getelementptr i8, ptr %src, i64 %offset
  %val = load i8, ptr %gep, align 1
  store i8 %val, ptr %dst, align 1
  %iv.next = add i16 %iv, 1
  %cmp = icmp ne i16 %iv.next, 0
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
