; RUN: opt  -passes='loop-load-elim,indvars,loop-distribute' -enable-loop-distribute %s

; REQUIRES: asserts
; XFAIL: *

define void @test_pr50940(ptr %A, ptr %B) {
entry:
  %gep.A.1 = getelementptr inbounds i16, ptr %A, i64 1
  br label %outer.header

outer.header:
  %gep.A.2 = getelementptr inbounds i16, ptr %gep.A.1, i64 1
  br i1 false, label %outer.latch, label %inner.ph

inner.ph:                             ; preds = %for.body5
  %lcssa.gep = phi ptr [ %gep.A.2, %outer.header ]
  %gep.A.3 = getelementptr inbounds i16, ptr %A, i64 3
  br label %inner

inner:
  %iv = phi i16 [ 0, %inner.ph ], [ %iv.next, %inner ]
  %l = load <2 x i16>, ptr %lcssa.gep, align 1
  store i16 0, ptr %gep.A.3, align 1
  store i16 1, ptr %B, align 1
  %iv.next = add nuw nsw i16 %iv, 1
  %c.1 = icmp ult i16 %iv, 38
  br i1 %c.1, label %inner, label %exit

outer.latch:
  br label %outer.header

exit:
  ret void
}
