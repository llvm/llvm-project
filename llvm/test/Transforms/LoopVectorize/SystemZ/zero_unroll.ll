; RUN: opt -S -passes=loop-vectorize -mtriple=s390x-linux-gnu -vectorizer-min-trip-count=8 < %s | FileCheck %s

define i32 @main(i32 %arg, ptr nocapture readnone %arg1) #0 {
;CHECK: vector.body:
entry:
  %0 = alloca i8, align 1
  br label %loop

loop:
  %storemerge.i.i = phi i8 [ 0, %entry ], [ %tmp12.i.i, %loop ]
  store i8 %storemerge.i.i, ptr %0, align 2
  %tmp8.i.i = icmp ult i8 %storemerge.i.i, 8
  %tmp12.i.i = add nuw nsw i8 %storemerge.i.i, 1
  br i1 %tmp8.i.i, label %loop, label %ret

ret:
  ret i32 0
}

attributes #0 = { "target-cpu"="z13" }

