; RUN: opt < %s -passes=loop-vectorize -S | FileCheck %s

; getDemandedBits() is called on the pointer-typed GEP instruction here.
; Only make sure we do not crash.

; CHECK: @test
define void @test(ptr %ptr, ptr %ptr_end) {
start:
  br label %loop

loop:
  %ptr2 = phi ptr [ %ptr3, %loop ], [ %ptr, %start ]
  %x = sext i8 undef to i64
  %ptr3 = getelementptr inbounds i8, ptr %ptr2, i64 1
  %cmp = icmp ult ptr %ptr3, %ptr_end
  br i1 %cmp, label %loop, label %end

end:
  ret void
}
