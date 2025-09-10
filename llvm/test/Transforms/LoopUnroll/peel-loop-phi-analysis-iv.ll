; RUN: opt -passes=loop-unroll -enable-peeling-for-iv -disable-output \
; RUN:     -pass-remarks-output=- %s | FileCheck %s
; RUN: opt -passes=loop-unroll-full -enable-peeling-for-iv -disable-output \
; RUN:     -pass-remarks-output=- %s | FileCheck %s

; void g(int);
declare void @g(i32)

; Check that phi analysis can handle a binary operator with an addition or a
; subtraction on loop-invariants or IVs. In the following case, the phis for x,
; y, a, and b become IVs by peeling.
;
;
; void g(int);
; void binary() {
;   int x = 0;
;   int y = 0;
;   int a = 42;
;   int b = 314;
;   for(int i = 0; i <100000; ++i) {
;     g(x);
;     g(b);
;     x = y;
;     y = a + 1;
;     a = i - 2;
;     b = i + a;
;   }
; }
;
;
; Consider the calls to g:
;
;                |   i |    g(x) |     g(b) |   x |   y |   a |     b |
; ---------------|-----|---------|----------|-----|-----|-----|-------|
;  1st iteration |   0 |    g(0) |   g(314) |   0 |  43 |  -2 |    -2 |
;  2nd iteration |   1 |    g(0) |    g(-2) |  43 |  -1 |  -1 |     0 |
;  3rd iteration |   2 |   g(43) |     g(0) |  -1 |   0 |   0 |     2 |
;  4th iteration |   3 |   g(-1) |     g(2) |   0 |   1 |   1 |     4 |
;  5th iteration |   4 |    g(0) |     g(4) |   1 |   2 |   2 |     6 |
; i-th iteration |   i |  g(i-5) | g(2*i-4) | i-4 | i-3 | i-2 | 2*i-4 |
;
; After the 4th iteration, the arguments to g become IVs, so peeling 3 times
; converts all PHIs into IVs.
;

; CHECK:      --- !Passed
; CHECK-NEXT: Pass:            loop-unroll
; CHECK-NEXT: Name:            Peeled
; CHECK-NEXT: Function:        binary_induction
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          ' peeled loop by '
; CHECK-NEXT:   - PeelCount:       '3'
; CHECK-NEXT:   - String:          ' iterations'
; CHECK-NEXT: ...
define void @binary_induction() {
entry:
  br label %for.body

exit:
  ret void

for.body:
  %i = phi i32 [ 0, %entry ], [ %i.next, %for.body ]
  %x = phi i32 [ 0, %entry ], [ %y, %for.body ]
  %y = phi i32 [ 0, %entry ], [ %y.next, %for.body ]
  %a = phi i32 [ 42, %entry ], [ %a.next, %for.body ]
  %b = phi i32 [ 314, %entry ], [ %b.next, %for.body ]
  tail call void @g(i32 %x)
  tail call void @g(i32 %b)
  %i.next = add i32 %i, 1
  %y.next = add i32 %a, 1
  %a.next = sub i32 %i, 2
  %b.next = add i32 %i, %a
  %cmp = icmp ne i32 %i.next, 100000
  br i1 %cmp, label %for.body, label %exit
}

; Check that peeling works fine in the following case. This is based on TSVC
; s291, where peeling 1 time makes the variable im an IV so we can vectorize
; the loop.
;
; #define N 100
; void f(float * restrict a, float * restrict b) {
;   int im = N - 1;
;   for (int i = 0; i < N; i++) {
;     a[i] = b[i] + b[im];
;     im = i;
;   }
; }
;

; CHECK:      --- !Passed
; CHECK-NEXT: Pass:            loop-unroll
; CHECK-NEXT: Name:            Peeled
; CHECK-NEXT: Function:        tsvc_s291
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          ' peeled loop by '
; CHECK-NEXT:   - PeelCount:       '1'
; CHECK-NEXT:   - String:          ' iterations'
; CHECK-NEXT: ...
define void @tsvc_s291(ptr noalias %a, ptr noalias %b) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [0, %entry], [ %i.next, %for.body ]
  %im = phi i32 [ 99, %entry ], [ %i, %for.body ]
  %a.idx = getelementptr inbounds float, ptr %a, i32 %i
  %b.idx.0 = getelementptr inbounds float, ptr %b, i32 %i
  %b.idx.1 = getelementptr inbounds float, ptr %b, i32 %im
  %lhs = load float, ptr %b.idx.0, align 4
  %rhs = load float, ptr %b.idx.1, align 4
  %sum = fadd float %lhs, %rhs
  store float %sum, ptr %a.idx, align 4
  %i.next = add i32 %i, 1
  %cmp = icmp ne i32 %i.next, 100
  br i1 %cmp, label %for.body, label %exit

exit:
  ret void
}

; Check that the unnecessary peeling occurs in the following case. The cause is
; that the analyzer determines a casted IV as a non-IV.
;
; for (unsigned int i=0; i<10000; i++)
;   a[(unsigned long)j] = 10;
;

; CHECK:      --- !Passed
; CHECK-NEXT: Pass:            loop-unroll
; CHECK-NEXT: Name:            Peeled
; CHECK-NEXT: Function:        induction_undesirable_peel1
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          ' peeled loop by '
; CHECK-NEXT:   - PeelCount:       '1'
; CHECK-NEXT:   - String:          ' iterations'
; CHECK-NEXT: ...
define void @induction_undesirable_peel1(ptr %a) {
entry:
  br label %for.body

for.body:
  %conv = phi i64 [ 0, %entry ], [ %conv.next, %for.body ]
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds nuw i32, ptr %a, i64 %conv
  store i32 10, ptr %arrayidx, align 4
  %iv.next = add nsw nuw i32 %iv, 1
  %conv.next = zext i32 %iv.next to i64
  %cmp = icmp ugt i64 10000, %conv.next
  br i1 %cmp, label %for.body, label %exit

exit:
  ret void
}

; Check that the unnecessary peeling occurs in the following case. The analyzer
; cannot detect that the difference between the initial value of %i and %j is
; equal to the step value of the %i.
;
; int j = 0;
; for (int i=1; i<N; i++) {
;   a[j] = 10;
;   j = i;
; }

; CHECK:      --- !Passed
; CHECK-NEXT: Pass:            loop-unroll
; CHECK-NEXT: Name:            Peeled
; CHECK-NEXT: Function:        induction_undesirable_peel2
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          ' peeled loop by '
; CHECK-NEXT:   - PeelCount:       '1'
; CHECK-NEXT:   - String:          ' iterations'
; CHECK-NEXT: ...
define void @induction_undesirable_peel2(ptr %a) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 1, %entry ], [ %i.next, %for.body ]
  %j = phi i32 [ 0, %entry ], [ %i, %for.body ]
  %arrayidx = getelementptr i32, ptr %a, i32 %j
  store i32 10, ptr %arrayidx, align 4
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i, 10000
  br i1 %cmp, label %for.body, label %exit

exit:
  ret void
}

; Check that phi analysis can handle a binary operator with an addition or a
; subtraction on loop-invariants or IVs. In the following case, the phis for x,
; y, a, and b become IVs by peeling.
;
;
; void g(int);
; void binary() {
;   int x = 2;
;   for(int i = 0; i <100000; ++i) {
;     g(x);
;     tmp = i - 2;
;     x = i - tmp;
;   }
; }
;
;
; Consider the calls to g:
;
;                | i | g(x) | tmp | x |
; ---------------|---|------|-----|---|
;  1st iteration | 0 | g(2) |  -2 | 2 |
;  2nd iteration | 1 | g(2) |  -1 | 2 |
;  3rd iteration | 2 | g(2) |   0 | 2 |
;  4th iteration | 3 | g(2) |   1 | 2 |
;
; In this case, the value of x is always 2. However, the analyzer recognizes
; the expression "i - tmp" as an IV.
;

; CHECK:      --- !Passed
; CHECK-NEXT: Pass:            loop-unroll
; CHECK-NEXT: Name:            Peeled
; CHECK-NEXT: Function:        induction_undesirable_peel3
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          ' peeled loop by '
; CHECK-NEXT:   - PeelCount:       '1'
; CHECK-NEXT:   - String:          ' iterations'
; CHECK-NEXT: ...
define void @induction_undesirable_peel3() {
entry:
  br label %for.body

exit:
  ret void

for.body:
  %i = phi i32 [ 0, %entry ], [ %i.next, %for.body ]
  %x = phi i32 [ 0, %entry ], [ %x.next, %for.body ]
  tail call void @g(i32 %x)
  %tmp = sub i32 %i, 2
  %x.next = sub i32 %i, %tmp
  %i.next = add i32 %i, 1
  %cmp = icmp ne i32 %i.next, 100000
  br i1 %cmp, label %for.body, label %exit
}
