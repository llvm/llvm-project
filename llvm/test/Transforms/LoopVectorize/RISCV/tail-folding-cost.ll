; REQUIRES: asserts
; RUN: opt < %s -passes=loop-vectorize -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN:   -mtriple riscv64-linux-gnu -mattr=+v,+f -S -disable-output -debug-only=loop-vectorize 2>&1 | FileCheck %s

; RUN: opt < %s -passes=loop-vectorize -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN:   -mtriple riscv64-linux-gnu -force-tail-folding-style=data-with-evl -mattr=+v,+f -S \
; RUN:   -disable-output -debug-only=loop-vectorize 2>&1 | FileCheck %s --check-prefix=EVL

; CHECK: Cost of 2 for VF 2: EMIT{{.*}} = active lane mask
; CHECK: Cost of 4 for VF 4: EMIT{{.*}} = active lane mask
; CHECK: Cost of 8 for VF 8: EMIT{{.*}} = active lane mask
; CHECK: Cost of 2 for VF vscale x 1: EMIT{{.*}} = active lane mask
; CHECK: Cost of 4 for VF vscale x 2: EMIT{{.*}} = active lane mask
; CHECK: Cost of 8 for VF vscale x 4: EMIT{{.*}} = active lane mask

; EVL: Cost of 1 for VF vscale x 1: EMIT{{.*}} = EXPLICIT-VECTOR-LENGTH
; EVL: Cost of 1 for VF vscale x 2: EMIT{{.*}} = EXPLICIT-VECTOR-LENGTH
; EVL: Cost of 1 for VF vscale x 4: EMIT{{.*}} = EXPLICIT-VECTOR-LENGTH

define void @simple_memset(i32 %val, ptr %ptr, i64 %n) #0 {
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %index = phi i64 [ %index.next, %while.body ], [ 0, %entry ]
  %gep = getelementptr i32, ptr %ptr, i64 %index
  store i32 %val, ptr %gep
  %index.next = add nsw i64 %index, 1
  %cmp10 = icmp ult i64 %index.next, %n
  br i1 %cmp10, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  ret void
}
