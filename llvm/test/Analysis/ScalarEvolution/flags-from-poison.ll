; RUN: opt < %s -S -analyze -scalar-evolution | FileCheck %s

; Positive and negative tests for inferring flags like nsw from
; reasoning about how a poison value from overflow would trigger
; undefined behavior.

define void @foo() {
  ret void
}

; Example where an add should get the nsw flag, so that a sext can be
; distributed over the add.
define void @test-add-nsw(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-nsw
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nsw>
  %index32 = add nsw i32 %i, %offset

; CHECK: %index64 =
; CHECK: --> {(sext i32 %offset to i64),+,1}<nsw>
  %index64 = sext i32 %index32 to i64

  %ptr = getelementptr inbounds float, float* %input, i64 %index64
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  call void @foo()
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Example where an add should get the nuw flag.
define void @test-add-nuw(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-nuw
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nuw>
  %index32 = add nuw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nuw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @test-add-nuw-from-icmp(float* %input, i32 %offset,
                                    i32 %numIterations) {
; CHECK-LABEL: @test-add-nuw-from-icmp
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nuw>
  %index32 = add nuw i32 %i, %offset
  %cmp = icmp sgt i32 %index32, 0
  %cmp.idx = sext i1 %cmp to i32

  %ptr = getelementptr inbounds float, float* %input, i32 %cmp.idx
  %nexti = add nuw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; With no load to trigger UB from poison, we cannot infer nsw.
define void @test-add-no-load(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-no-load
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nuw i32 %i, 1
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; The current code is only supposed to look at the loop header, so
; it should not infer nsw in this case, as that would require looking
; outside the loop header.
define void @test-add-not-header(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-not-header
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop2 ], [ 0, %entry ]
  br label %loop2
loop2:

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Same thing as test-add-not-header, but in this case only the load
; instruction is outside the loop header.
define void @test-add-not-header2(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-not-header2
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop2 ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nsw>
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  br label %loop2
loop2:
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Similar to test-add-not-header, but in this case the load
; instruction may not be executed.
define void @test-add-not-header3(float* %input, i32 %offset, i32 %numIterations,
                                 i1* %cond_buf) {
; CHECK-LABEL: @test-add-not-header3
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop2 ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  %cond = load volatile i1, i1* %cond_buf
  br i1 %cond, label %loop2, label %exit
loop2:
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Same thing as test-add-not-header2, except we have a few extra
; blocks.
define void @test-add-not-header4(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-not-header4
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop2 ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nsw>
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  br label %loop3
loop3:
  br label %loop4
loop4:
  br label %loop2
loop2:
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Demonstrate why we need a Visited set in llvm::isKnownNotFullPoison.
define void @test-add-not-header5(float* %input, i32 %offset) {
; CHECK-LABEL: @test-add-not-header5
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  br label %loop

exit:
  ret void
}

; The call instruction makes it not guaranteed that the add will be
; executed, since it could run forever or throw an exception, so we
; cannot assume that the UB is realized.
define void @test-add-call(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-call
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  call void @foo()
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Same issue as test-add-call, but this time the call is between the
; producer of poison and the load that consumes it.
define void @test-add-call2(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-call2
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  call void @foo()
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Without inbounds, GEP does not propagate poison in the very
; conservative approach used here.
define void @test-add-no-inbounds(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-no-inbounds
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Multiplication by a non-zero constant propagates poison if there is
; a nuw or nsw flag on the multiplication.
define void @test-add-mul-propagates(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-mul-propagates
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nsw>
  %index32 = add nsw i32 %i, %offset

  %indexmul = mul nuw i32 %index32, 2
  %ptr = getelementptr inbounds float, float* %input, i32 %indexmul
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Multiplication by a non-constant should not propagate poison in the
; very conservative approach used here.
define void @test-add-mul-no-propagation(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-mul-no-propagation
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  %index32 = add nsw i32 %i, %offset

  %indexmul = mul nsw i32 %index32, %offset
  %ptr = getelementptr inbounds float, float* %input, i32 %indexmul
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Multiplication by a non-zero constant does not propagate poison
; without a no-wrap flag.
define void @test-add-mul-no-propagation2(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-mul-no-propagation2
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nw>
  %index32 = add nsw i32 %i, %offset

  %indexmul = mul i32 %index32, 2
  %ptr = getelementptr inbounds float, float* %input, i32 %indexmul
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Division by poison triggers UB.
define void @test-add-div(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-div
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %j =
; CHECK: --> {%offset,+,1}<nsw>
  %j = add nsw i32 %i, %offset

  %q = sdiv i32 %numIterations, %j
  %nexti = add nsw i32 %i, 1
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Remainder of poison by non-poison divisor does not trigger UB.
define void @test-add-div2(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-div2
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %j =
; CHECK: --> {%offset,+,1}<nw>
  %j = add nsw i32 %i, %offset

  %q = sdiv i32 %j, %numIterations
  %nexti = add nsw i32 %i, 1
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Store to poison address triggers UB.
define void @test-add-store(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-store
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {%offset,+,1}<nsw>
  %index32 = add nsw i32 %i, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  store float 1.0, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Three sequential adds where the middle add should have nsw. There is
; a special case for sequential adds and this test covers that. We have to
; put the final add first in the program since otherwise the special case
; is not triggered, hence the strange basic block ordering.
define void @test-add-twice(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-add-twice
entry:
  br label %loop
loop2:
; CHECK: %seq =
; CHECK: --> {(2 + %offset),+,1}<nw>
  %seq = add nsw nuw i32 %index32, 1
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

loop:
  %i = phi i32 [ %nexti, %loop2 ], [ 0, %entry ]

  %j = add nsw i32 %i, 1
; CHECK: %index32 =
; CHECK: --> {(1 + %offset)<nsw>,+,1}<nsw>
  %index32 = add nsw i32 %j, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  store float 1.0, float* %ptr, align 4
  br label %loop2
exit:
  ret void
}

; Example where a mul should get the nsw flag, so that a sext can be
; distributed over the mul.
define void @test-mul-nsw(float* %input, i32 %stride, i32 %numIterations) {
; CHECK-LABEL: @test-mul-nsw
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {0,+,%stride}<nsw>
  %index32 = mul nsw i32 %i, %stride

; CHECK: %index64 =
; CHECK: --> {0,+,(sext i32 %stride to i64)}<nsw>
  %index64 = sext i32 %index32 to i64

  %ptr = getelementptr inbounds float, float* %input, i64 %index64
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Example where a mul should get the nuw flag.
define void @test-mul-nuw(float* %input, i32 %stride, i32 %numIterations) {
; CHECK-LABEL: @test-mul-nuw
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {0,+,%stride}<nuw>
  %index32 = mul nuw i32 %i, %stride

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nuw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; Example where a shl should get the nsw flag, so that a sext can be
; distributed over the shl.
define void @test-shl-nsw(float* %input, i32 %start, i32 %numIterations) {
; CHECK-LABEL: @test-shl-nsw
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ %start, %entry ]

; CHECK: %index32 =
; CHECK: --> {(256 * %start),+,256}<nsw>
  %index32 = shl nsw i32 %i, 8

; CHECK: %index64 =
; CHECK: --> {(sext i32 (256 * %start) to i64),+,256}<nsw>
  %index64 = sext i32 %index32 to i64

  %ptr = getelementptr inbounds float, float* %input, i64 %index64
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Example where a shl should get the nuw flag.
define void @test-shl-nuw(float* %input, i32 %numIterations) {
; CHECK-LABEL: @test-shl-nuw
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {0,+,512}<nuw>
  %index32 = shl nuw i32 %i, 9

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nuw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; Example where a sub should *not* get the nsw flag, because of how
; scalar evolution represents A - B as A + (-B) and -B can wrap even
; in cases where A - B does not.
define void @test-sub-no-nsw(float* %input, i32 %start, i32 %sub, i32 %numIterations) {
; CHECK-LABEL: @test-sub-no-nsw
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ %start, %entry ]

; CHECK: %index32 =
; CHECK: --> {((-1 * %sub) + %start),+,1}<nw>
  %index32 = sub nsw i32 %i, %sub
  %index64 = sext i32 %index32 to i64

  %ptr = getelementptr inbounds float, float* %input, i64 %index64
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Example where a sub should get the nsw flag as the RHS cannot be the
; minimal signed value.
define void @test-sub-nsw(float* %input, i32 %start, i32 %sub, i32 %numIterations) {
; CHECK-LABEL: @test-sub-nsw
entry:
  %halfsub = ashr i32 %sub, 1
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ %start, %entry ]

; CHECK: %index32 =
; CHECK: --> {((-1 * %halfsub)<nsw> + %start)<nsw>,+,1}<nsw>
  %index32 = sub nsw i32 %i, %halfsub
  %index64 = sext i32 %index32 to i64

  %ptr = getelementptr inbounds float, float* %input, i64 %index64
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Example where a sub should get the nsw flag, since the LHS is non-negative,
; which implies that the RHS cannot be the minimal signed value.
define void @test-sub-nsw-lhs-non-negative(float* %input, i32 %sub, i32 %numIterations) {
; CHECK-LABEL: @test-sub-nsw-lhs-non-negative
entry:
  br label %loop
loop:
  %i = phi i32 [ %nexti, %loop ], [ 0, %entry ]

; CHECK: %index32 =
; CHECK: --> {(-1 * %sub),+,1}<nsw>
  %index32 = sub nsw i32 %i, %sub

; CHECK: %index64 =
; CHECK: --> {(sext i32 (-1 * %sub) to i64),+,1}<nsw>
  %index64 = sext i32 %index32 to i64

  %ptr = getelementptr inbounds float, float* %input, i64 %index64
  %nexti = add nsw i32 %i, 1
  %f = load float, float* %ptr, align 4
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop
exit:
  ret void
}

; Two adds with a sub in the middle and the sub should have nsw. There is
; a special case for sequential adds/subs and this test covers that. We have to
; put the final add first in the program since otherwise the special case
; is not triggered, hence the strange basic block ordering.
define void @test-sub-with-add(float* %input, i32 %offset, i32 %numIterations) {
; CHECK-LABEL: @test-sub-with-add
entry:
  br label %loop
loop2:
; CHECK: %seq =
; CHECK: --> {(2 + (-1 * %offset)),+,1}<nw>
  %seq = add nsw nuw i32 %index32, 1
  %exitcond = icmp eq i32 %nexti, %numIterations
  br i1 %exitcond, label %exit, label %loop

loop:
  %i = phi i32 [ %nexti, %loop2 ], [ 0, %entry ]

  %j = add nsw i32 %i, 1
; CHECK: %index32 =
; CHECK: --> {(1 + (-1 * %offset))<nsw>,+,1}<nsw>
  %index32 = sub nsw i32 %j, %offset

  %ptr = getelementptr inbounds float, float* %input, i32 %index32
  %nexti = add nsw i32 %i, 1
  store float 1.0, float* %ptr, align 4
  br label %loop2
exit:
  ret void
}


; Subtraction of two recurrences. The addition in the SCEV that this
; maps to is NSW, but the negation of the RHS does not since that
; recurrence could be the most negative representable value.
define void @subrecurrences(i32 %outer_l, i32 %inner_l, i32 %val) {
; CHECK-LABEL: @subrecurrences
 entry:
  br label %outer

outer:
  %o_idx = phi i32 [ 0, %entry ], [ %o_idx.inc, %outer.be ]
  %o_idx.inc = add nsw i32 %o_idx, 1
  %cond = icmp eq i32 %o_idx, %val
  br i1 %cond, label %inner, label %outer.be

inner:
  %i_idx = phi i32 [ 0, %outer ], [ %i_idx.inc, %inner ]
  %i_idx.inc = add nsw i32 %i_idx, 1
; CHECK: %v =
; CHECK-NEXT: --> {{[{][{]}}-1,+,-1}<nw><%outer>,+,1}<nsw><%inner>
  %v = sub nsw i32 %i_idx, %o_idx.inc
  %forub = udiv i32 1, %v
  %cond2 = icmp eq i32 %i_idx, %inner_l
  br i1 %cond2, label %outer.be, label %inner

outer.be:
  %cond3 = icmp eq i32 %o_idx, %outer_l
  br i1 %cond3, label %exit, label %outer

exit:
  ret void
}


; PR28932: Don't assert on non-SCEV-able value %2.
%struct.anon = type { i8* }
@a = common global %struct.anon* null, align 8
@b = common global i32 0, align 4
declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32)
declare void @llvm.trap()
define i32 @pr28932() {
entry:
  %.pre = load %struct.anon*, %struct.anon** @a, align 8
  %.pre7 = load i32, i32* @b, align 4
  br label %for.cond

for.cond:                                         ; preds = %cont6, %entry
  %0 = phi i32 [ %3, %cont6 ], [ %.pre7, %entry ]
  %1 = phi %struct.anon* [ %.ph, %cont6 ], [ %.pre, %entry ]
  %tobool = icmp eq %struct.anon* %1, null
  %2 = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %0, i32 1)
  %3 = extractvalue { i32, i1 } %2, 0
  %4 = extractvalue { i32, i1 } %2, 1
  %idxprom = sext i32 %3 to i64
  %5 = getelementptr inbounds %struct.anon, %struct.anon* %1, i64 0, i32 0
  %6 = load i8*, i8** %5, align 8
  %7 = getelementptr inbounds i8, i8* %6, i64 %idxprom
  %8 = load i8, i8* %7, align 1
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %for.cond
  br i1 %4, label %trap, label %cont6

trap:                                             ; preds = %if.else, %if.then
  tail call void @llvm.trap()
  unreachable

if.else:                                          ; preds = %for.cond
  br i1 %4, label %trap, label %cont1

cont1:                                            ; preds = %if.else
  %conv5 = sext i8 %8 to i64
  %9 = inttoptr i64 %conv5 to %struct.anon*
  store %struct.anon* %9, %struct.anon** @a, align 8
  br label %cont6

cont6:                                            ; preds = %cont1, %if.then
  %.ph = phi %struct.anon* [ %9, %cont1 ], [ %1, %if.then ]
  store i32 %3, i32* @b, align 4
  br label %for.cond
}
