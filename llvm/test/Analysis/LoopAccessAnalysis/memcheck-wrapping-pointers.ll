; RUN: opt -passes='print<access-info>' %s -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

; i and i + 1 can overflow in the following kernel:
; void test1(unsigned long long x, int *a, int *b) {
;  for (unsigned i = 0; i < x; ++i)
;    b[i] = a[i+1] + 1;
; }
;
; If accesses to a and b can alias, we need to emit a run-time alias check
; between accesses to a and b. However, when i and i + 1 can wrap, their
; SCEV expression is not an AddRec. We need to create SCEV predicates and
; coerce the expressions to AddRecs in order to be able to emit the run-time
; alias check.
;
; The accesses at b[i] and a[i+1] correspond to the addresses %arrayidx and
; %arrayidx4 in the test. The SCEV expressions for these are:
;  ((4 * (zext i32 {1,+,1}<%for.body> to i64))<nuw><nsw> + %a)<nsw>
;  ((4 * (zext i32 {0,+,1}<%for.body> to i64))<nuw><nsw> + %b)<nsw>
;
; The transformed expressions are:
;  i64 {(4 + %a),+,4}<%for.body>
;  i64 {(4 + %b),+,4}<%for.body>

; CHECK-LABEL: test1
; CHECK:         Memory dependences are safe with run-time checks
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Check 0:
; CHECK-NEXT:      Comparing group
; CHECK-NEXT:        %arrayidx4 = getelementptr inbounds i32, ptr %b, i64 %conv11
; CHECK-NEXT:      Against group
; CHECK-NEXT:        %arrayidx = getelementptr inbounds i32, ptr %a, i64 %idxprom
; CHECK-NEXT:    Grouped accesses:
; CHECK-NEXT:      Group
; CHECK-NEXT:        (Low: %b High: ((4 * (1 umax %x)) + %b))
; CHECK-NEXT:          Member: {%b,+,4}<%for.body>
; CHECK-NEXT:      Group
; CHECK-NEXT:        (Low: (4 + %a) High: (4 + (4 * (1 umax %x)) + %a))
; CHECK-NEXT:          Member: {(4 + %a),+,4}<%for.body>
; CHECK:         Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:    SCEV assumptions:
; CHECK-NEXT:    {1,+,1}<%for.body> Added Flags: <nusw>
; CHECK:         Expressions re-written:
; CHECK-NEXT:    [PSE]  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %idxprom:
; CHECK-NEXT:      ((4 * (zext i32 {1,+,1}<%for.body> to i64))<nuw><nsw> + %a)<nuw>
; CHECK-NEXT:      --> {(4 + %a),+,4}<%for.body>
; CHECK-NEXT:    [PSE]  %arrayidx4 = getelementptr inbounds i32, ptr %b, i64 %conv11:
; CHECK-NEXT:      ((4 * (zext i32 {0,+,1}<%for.body> to i64))<nuw><nsw> + %b)<nuw>
; CHECK-NEXT:      --> {%b,+,4}<%for.body>
define void @test1(i64 %x, ptr %a, ptr %b) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %conv11 = phi i64 [ %conv, %for.body ], [ 0, %entry ]
  %i.010 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %add = add i32 %i.010, 1
  %idxprom = zext i32 %add to i64
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %idxprom
  %ld = load i32, ptr %arrayidx, align 4
  %add2 = add nsw i32 %ld, 1
  %arrayidx4 = getelementptr inbounds i32, ptr %b, i64 %conv11
  store i32 %add2, ptr %arrayidx4, align 4
  %conv = zext i32 %add to i64
  %cmp = icmp ult i64 %conv, %x
  br i1 %cmp, label %for.body, label %exit

exit:
  ret void
}

; i can overflow in the following kernel:
; void test2(unsigned long long x, int *a) {
;   for (unsigned i = 0; i < x; ++i)
;     a[i] = a[i] + 1;
; }
;
; We need to check that i doesn't wrap, but we don't need a run-time alias
; check. We also need an extra no-wrap check to get the backedge taken count.

; CHECK-LABEL: test2
; CHECK: Memory dependences are safe
; CHECK: SCEV assumptions:
; CHECK-NEXT:   {1,+,1}<%for.body> Added Flags: <nusw>
 define void @test2(i64 %x, ptr %a) {
entry:
  br label %for.body

for.body:
  %conv11 = phi i64 [ %conv, %for.body ], [ 0, %entry ]
  %i.010 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %conv11
  %ld = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %ld, 1
  store i32 %add, ptr %arrayidx, align 4
  %inc = add i32 %i.010, 1
  %conv = zext i32 %inc to i64
  %cmp = icmp ult i64 %conv, %x
  br i1 %cmp, label %for.body, label %exit

exit:
  ret void
}
