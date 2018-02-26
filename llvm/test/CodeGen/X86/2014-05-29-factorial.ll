; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s
; CHECK: decq [[X:%rdi|%rcx]]
; CHECK-NOT: testq [[X]], [[X]]

define i64 @fact2(i64 %x) {
entry:
  br label %while.body

while.body:
  %result.06 = phi i64 [ %mul, %while.body ], [ 1, %entry ]
  %x.addr.05 = phi i64 [ %dec, %while.body ], [ %x, %entry ]
  %mul = mul nsw i64 %result.06, %x.addr.05
  %dec = add nsw i64 %x.addr.05, -1
  %cmp = icmp sgt i64 %dec, 0
  br i1 %cmp, label %while.body, label %while.end.loopexit

while.end.loopexit:
  %mul.lcssa = phi i64 [ %mul, %while.body ]
  br label %while.end

while.end:
  %result.0.lcssa = phi i64 [ %mul.lcssa, %while.end.loopexit ]
  ret i64 %result.0.lcssa
}
