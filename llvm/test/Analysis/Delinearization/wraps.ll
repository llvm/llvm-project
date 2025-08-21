; RUN: opt < %s -passes='print<delinearization>' -disable-output 2>&1 | FileCheck %s

; In the following case, we don't know the concret value of `m`, so we cannot
; deduce no-wrap behavior for the quotient/remainder divided by `m`. However,
; we can infer `{0,+,1}<%loop>` is nuw and nsw from the induction variable.
;
; for (int i = 0; i < btc; i++)
;   a[i * (m + 42)] = 0;

; CHECK:      AccessFunction: {0,+,(42 + %m)}<nuw><nsw><%loop>
; CHECK-NEXT: Base offset: %a
; CHECK-NEXT: ArrayDecl[UnknownSize][%m] with elements of 1 bytes.
; CHECK-NEXT: ArrayRef[{0,+,1}<nuw><nsw><%loop>][{0,+,42}<%loop>]
define void @divide_by_m0(ptr %a, i64 %m, i64 %btc) {
entry:
  %stride = add nsw nuw i64 %m, 42
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %offset = phi i64 [ 0, %entry ], [ %offset.next, %loop ]
  %idx = getelementptr inbounds i8, ptr %a, i64 %offset
  store i8 0, ptr %idx
  %i.next = add nsw nuw i64 %i, 1
  %offset.next = add nsw nuw i64 %offset, %stride
  %cond = icmp eq i64 %i.next, %btc
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

; In the following case, we don't know the concret value of `m`, so we cannot
; deduce no-wrap behavior for the quotient/remainder divided by `m`. Also, we
; don't infer nsw/nuw from the induction variable in this case.
;
; for (int i = 0; i < btc; i++)
;   a[i * (2 * m + 42)] = 0;

; CHECK:      AccessFunction: {0,+,(42 + (2 * %m))}<nuw><nsw><%loop>
; CHECK-NEXT: Base offset: %a
; CHECK-NEXT: ArrayDecl[UnknownSize][%m] with elements of 1 bytes.
; CHECK-NEXT: ArrayRef[{0,+,2}<%loop>][{0,+,42}<%loop>]
define void @divide_by_m2(ptr %a, i64 %m, i64 %btc) {
entry:
  %m2 = add nsw nuw i64 %m, %m
  %stride = add nsw nuw i64 %m2, 42
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %offset = phi i64 [ 0, %entry ], [ %offset.next, %loop ]
  %idx = getelementptr inbounds i8, ptr %a, i64 %offset
  store i8 0, ptr %idx
  %i.next = add nsw nuw i64 %i, 1
  %offset.next = add nsw nuw i64 %offset, %stride
  %cond = icmp eq i64 %i.next, %btc
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

; In the following case, the `i * 2 * d` is always zero, so it's nsw and nuw.
; However, the quotient divided by `d` is neither nsw nor nuw.
;
; if (d == 0)
;   for (unsigned long long i = 0; i != UINT64_MAX; i++)
;     a[i * 2 * d] = 42;

; CHECK:      AccessFunction: {0,+,(2 * %d)}<nuw><nsw><%loop>
; CHECK-NEXT: Base offset: %a
; CHECK-NEXT: ArrayDecl[UnknownSize][%d] with elements of 1 bytes.
; CHECK-NEXT: ArrayRef[{0,+,2}<%loop>][0]
define void @divide_by_zero(ptr %a, i64 %d) {
entry:
  %guard = icmp eq i64 %d, 0
  br i1 %guard, label %loop.preheader, label %exit

loop.preheader:
  %stride = mul nsw nuw i64 %d, 2  ; since %d is 0, %stride is also 0
  br label %loop

loop:
  %i = phi i64 [ 0, %loop.preheader ], [ %i.next, %loop ]
  %offset = phi i64 [ 0, %loop.preheader ], [ %offset.next, %loop ]
  %idx = getelementptr inbounds i8, ptr %a, i64 %offset
  store i8 42, ptr %idx
  %i.next = add nuw i64 %i, 1
  %offset.next = add nsw nuw i64 %offset, %stride
  %cond = icmp eq i64 %i.next, -1
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

; In the following case, the `i * (d + 1)` is always zero, so it's nsw and nuw.
; However, the quotient/remainder divided by `d` is not nsw.
;
; if (d == UINT64_MAX)
;   for (unsigned long long i = 0; i != d; i++)
;     a[i * (d + 1)] = 42;

; CHECK:      AccessFunction: {0,+,(1 + %d)}<nuw><nsw><%loop>
; CHECK-NEXT: Base offset: %a
; CHECK-NEXT: ArrayDecl[UnknownSize][%d] with elements of 1 bytes.
; CHECK-NEXT: ArrayRef[{0,+,1}<nuw><%loop>][{0,+,1}<nuw><%loop>]
define void @divide_by_minus_one(ptr %a, i64 %d) {
entry:
  %guard = icmp eq i64 %d, -1
  br i1 %guard, label %loop.preheader, label %exit

loop.preheader:
  %stride = add nsw i64 %d, 1  ; since %d is -1, %stride is 0
  br label %loop

loop:
  %i = phi i64 [ 0, %loop.preheader ], [ %i.next, %loop ]
  %offset = phi i64 [ 0, %loop.preheader ], [ %offset.next, %loop ]
  %idx = getelementptr inbounds i8, ptr %a, i64 %offset
  store i8 42, ptr %idx
  %i.next = add nuw i64 %i, 1
  %offset.next = add nsw nuw i64 %offset, %stride
  %cond = icmp eq i64 %i.next, %d
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}
