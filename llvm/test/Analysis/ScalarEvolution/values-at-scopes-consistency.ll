; RUN: opt -S -passes='verify<scalar-evolution>' < %s

; Make sure this does not fail ValuesAtScopes consistency verification.
; This used to register a ValuesAtScopes user, even though nothing was
; added to ValuesAtScope due to a prior invalidation.

define void @main(i8* %p) {
entry:
  br label %loop1

loop1:
  br label %loop2

loop2:
  %i = phi i64 [ 0, %loop1 ], [ %i.next, %loop2.latch ]
  %i.next = add nuw nsw i64 %i, 1
  %gep = getelementptr i8, i8* %p, i64 %i
  %val = load i8, i8* %gep
  %c = icmp eq i8 %val, 0
  br i1 %c, label %loop2.latch, label %exit

loop2.latch:
  br i1 false, label %loop2, label %loop1.latch

loop1.latch:
  br label %loop1

exit:
  ret void
}
