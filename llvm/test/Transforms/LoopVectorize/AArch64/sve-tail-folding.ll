; RUN: opt -S -loop-vectorize -prefer-predicate-over-epilogue=predicate-dont-vectorize < %s | FileCheck %s

; CHECK-NOT: vector.body:

target triple = "aarch64-unknown-linux-gnu"

define void @tail_predication(i32 %init, i32* %ptr, i32 %val) #0 {
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %index = phi i32 [ %index.dec, %while.body ], [ %init, %entry ]
  %gep = getelementptr i32, i32* %ptr, i32 %index
  store i32 %val, i32* %gep
  %index.dec = add nsw i32 %index, -1
  %cmp10 = icmp sgt i32 %index, 0
  br i1 %cmp10, label %while.body, label %while.end.loopexit

while.end.loopexit:                               ; preds = %while.body
  ret void
}

attributes #0 = { "target-features"="+sve" }
