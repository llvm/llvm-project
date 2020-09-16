; RUN: opt -loop-accesses -analyze -enable-new-pm=0 < %s | FileCheck %s
; RUN: opt -passes='require<scalar-evolution>,require<aa>,loop(print-access-info)' -disable-output  < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnueabi"

; 3 reads and 3 writes should need 12 memchecks
; CHECK: function 'testf':
; CHECK: Memory dependences are safe with run-time checks

; Memory dependencies have labels starting from 0, so in
; order to verify that we have n checks, we look for
; (n-1): and not n:.

; CHECK: Run-time memory checks:
; CHECK-NEXT: Check 0:
; CHECK: Check 11:
; CHECK-NOT: Check 12:

define void @testf(i16* %a,
               i16* %b,
               i16* %c,
               i16* %d,
               i16* %e,
               i16* %f) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %add = add nuw nsw i64 %ind, 1

  %arrayidxA = getelementptr inbounds i16, i16* %a, i64 %ind
  %loadA = load i16, i16* %arrayidxA, align 2

  %arrayidxB = getelementptr inbounds i16, i16* %b, i64 %ind
  %loadB = load i16, i16* %arrayidxB, align 2

  %arrayidxC = getelementptr inbounds i16, i16* %c, i64 %ind
  %loadC = load i16, i16* %arrayidxC, align 2

  %mul = mul i16 %loadB, %loadA
  %mul1 = mul i16 %mul, %loadC

  %arrayidxD = getelementptr inbounds i16, i16* %d, i64 %ind
  store i16 %mul1, i16* %arrayidxD, align 2

  %arrayidxE = getelementptr inbounds i16, i16* %e, i64 %ind
  store i16 %mul, i16* %arrayidxE, align 2

  %arrayidxF = getelementptr inbounds i16, i16* %f, i64 %ind
  store i16 %mul1, i16* %arrayidxF, align 2

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; The following (testg and testh) check that we can group
; memory checks of accesses which differ by a constant value.
; Both tests are based on the following C code:
;
; void testh(short *a, short *b, short *c) {
;   unsigned long ind = 0;
;   for (unsigned long ind = 0; ind < 20; ++ind) {
;     c[2 * ind] = a[ind] * a[ind + 1];
;     c[2 * ind + 1] = a[ind] * a[ind + 1] * b[ind];
;   }
; }
;
; It is sufficient to check the intervals
; [a, a + 21], [b, b + 20] against [c, c + 41].

; 3 reads and 2 writes - two of the reads can be merged,
; and the writes can be merged as well. This gives us a
; total of 2 memory checks.

; CHECK: function 'testg':

; CHECK: Run-time memory checks:
; CHECK-NEXT:   Check 0:
; CHECK-NEXT:     Comparing group ([[ZERO:.+]]):
; CHECK-NEXT:       %arrayidxC1 = getelementptr inbounds i16, i16* %c, i64 %store_ind_inc
; CHECK-NEXT:       %arrayidxC = getelementptr inbounds i16, i16* %c, i64 %store_ind
; CHECK-NEXT:     Against group ([[ONE:.+]]):
; CHECK-NEXT:       %arrayidxA1 = getelementptr inbounds i16, i16* %a, i64 %add
; CHECK-NEXT:       %arrayidxA = getelementptr inbounds i16, i16* %a, i64 %ind
; CHECK-NEXT:   Check 1:
; CHECK-NEXT:     Comparing group ({{.*}}[[ZERO]]):
; CHECK-NEXT:       %arrayidxC1 = getelementptr inbounds i16, i16* %c, i64 %store_ind_inc
; CHECK-NEXT:       %arrayidxC = getelementptr inbounds i16, i16* %c, i64 %store_ind
; CHECK-NEXT:     Against group ([[TWO:.+]]):
; CHECK-NEXT:       %arrayidxB = getelementptr inbounds i16, i16* %b, i64 %ind
; CHECK-NEXT:   Grouped accesses:
; CHECK-NEXT:    Group {{.*}}[[ZERO]]:
; CHECK-NEXT:       (Low: %c High: (80 + %c))
; CHECK-NEXT:         Member: {(2 + %c)<nsw>,+,4}
; CHECK-NEXT:         Member: {%c,+,4}
; CHECK-NEXT:     Group {{.*}}[[ONE]]:
; CHECK-NEXT:       (Low: %a High: (42 + %a))
; CHECK-NEXT:         Member: {(2 + %a)<nsw>,+,2}
; CHECK-NEXT:         Member: {%a,+,2}
; CHECK-NEXT:     Group {{.*}}[[TWO]]:
; CHECK-NEXT:       (Low: %b High: (40 + %b))
; CHECK-NEXT:         Member: {%b,+,2}

define void @testg(i16* %a,
               i16* %b,
               i16* %c) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]
  %store_ind = phi i64 [ 0, %entry ], [ %store_ind_next, %for.body ]

  %add = add nuw nsw i64 %ind, 1
  %store_ind_inc = add nuw nsw i64 %store_ind, 1
  %store_ind_next = add nuw nsw i64 %store_ind_inc, 1

  %arrayidxA = getelementptr inbounds i16, i16* %a, i64 %ind
  %loadA = load i16, i16* %arrayidxA, align 2

  %arrayidxA1 = getelementptr inbounds i16, i16* %a, i64 %add
  %loadA1 = load i16, i16* %arrayidxA1, align 2

  %arrayidxB = getelementptr inbounds i16, i16* %b, i64 %ind
  %loadB = load i16, i16* %arrayidxB, align 2

  %mul = mul i16 %loadA, %loadA1
  %mul1 = mul i16 %mul, %loadB

  %arrayidxC = getelementptr inbounds i16, i16* %c, i64 %store_ind
  store i16 %mul1, i16* %arrayidxC, align 2

  %arrayidxC1 = getelementptr inbounds i16, i16* %c, i64 %store_ind_inc
  store i16 %mul, i16* %arrayidxC1, align 2

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; 3 reads and 2 writes - the writes can be merged into a single
; group, but the GEPs used for the reads are not marked as inbounds.
; We can still merge them because we are using a unit stride for
; accesses, so we cannot overflow the GEPs.

; CHECK: function 'testh':
; CHECK: Run-time memory checks:
; CHECK-NEXT:   Check 0:
; CHECK-NEXT:     Comparing group ([[ZERO:.+]]):
; CHECK-NEXT:         %arrayidxC1 = getelementptr inbounds i16, i16* %c, i64 %store_ind_inc
; CHECK-NEXT:         %arrayidxC = getelementptr inbounds i16, i16* %c, i64 %store_ind
; CHECK-NEXT:     Against group ([[ONE:.+]]):
; CHECK-NEXT:         %arrayidxA1 = getelementptr i16, i16* %a, i64 %add
; CHECK-NEXT:         %arrayidxA = getelementptr i16, i16* %a, i64 %ind
; CHECK-NEXT:   Check 1:
; CHECK-NEXT:     Comparing group ({{.*}}[[ZERO]]):
; CHECK-NEXT:         %arrayidxC1 = getelementptr inbounds i16, i16* %c, i64 %store_ind_inc
; CHECK-NEXT:         %arrayidxC = getelementptr inbounds i16, i16* %c, i64 %store_ind
; CHECK-NEXT:     Against group ([[TWO:.+]]):
; CHECK-NEXT:         %arrayidxB = getelementptr i16, i16* %b, i64 %ind
; CHECK-NEXT:   Grouped accesses:
; CHECK-NEXT:     Group {{.*}}[[ZERO]]:
; CHECK-NEXT:       (Low: %c High: (80 + %c))
; CHECK-NEXT:         Member: {(2 + %c)<nsw>,+,4}
; CHECK-NEXT:         Member: {%c,+,4}
; CHECK-NEXT:     Group {{.*}}[[ONE]]:
; CHECK-NEXT:       (Low: %a High: (42 + %a))
; CHECK-NEXT:         Member: {(2 + %a),+,2}
; CHECK-NEXT:         Member: {%a,+,2}
; CHECK-NEXT:     Group {{.*}}[[TWO]]:
; CHECK-NEXT:       (Low: %b High: (40 + %b))
; CHECK-NEXT:         Member: {%b,+,2}

define void @testh(i16* %a,
               i16* %b,
               i16* %c) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]
  %store_ind = phi i64 [ 0, %entry ], [ %store_ind_next, %for.body ]

  %add = add nuw nsw i64 %ind, 1
  %store_ind_inc = add nuw nsw i64 %store_ind, 1
  %store_ind_next = add nuw nsw i64 %store_ind_inc, 1

  %arrayidxA = getelementptr i16, i16* %a, i64 %ind
  %loadA = load i16, i16* %arrayidxA, align 2

  %arrayidxA1 = getelementptr i16, i16* %a, i64 %add
  %loadA1 = load i16, i16* %arrayidxA1, align 2

  %arrayidxB = getelementptr i16, i16* %b, i64 %ind
  %loadB = load i16, i16* %arrayidxB, align 2

  %mul = mul i16 %loadA, %loadA1
  %mul1 = mul i16 %mul, %loadB

  %arrayidxC = getelementptr inbounds i16, i16* %c, i64 %store_ind
  store i16 %mul1, i16* %arrayidxC, align 2

  %arrayidxC1 = getelementptr inbounds i16, i16* %c, i64 %store_ind_inc
  store i16 %mul, i16* %arrayidxC1, align 2

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; Don't merge pointers if we need to perform a check against a pointer
; to the same underlying object (doing so would emit a check that could be
; falsely invalidated) For example, in the following loop:
;
; for (i = 0; i < 5000; ++i)
;   a[i + offset] = a[i] + a[i + 10000]
;
; we should not merge the intervals associated with the reads (0,5000) and
; (10000, 15000) into (0, 15000) as this will pottentially fail the check
; against the interval associated with the write.
;
; We cannot have this check unless ShouldRetryWithRuntimeCheck is set,
; and therefore the grouping algorithm would create a separate group for
; each pointer.

; CHECK: function 'testi':
; CHECK: Run-time memory checks:
; CHECK-NEXT:   Check 0:
; CHECK-NEXT:     Comparing group ([[ZERO:.+]]):
; CHECK-NEXT:       %storeidx = getelementptr inbounds i16, i16* %a, i64 %store_ind
; CHECK-NEXT:     Against group ([[ONE:.+]]):
; CHECK-NEXT:       %arrayidxA1 = getelementptr i16, i16* %a, i64 %ind
; CHECK-NEXT:   Check 1:
; CHECK-NEXT:     Comparing group ({{.*}}[[ZERO]]):
; CHECK-NEXT:       %storeidx = getelementptr inbounds i16, i16* %a, i64 %store_ind
; CHECK-NEXT:     Against group ([[TWO:.+]]):
; CHECK-NEXT:       %arrayidxA2 = getelementptr i16, i16* %a, i64 %ind2
; CHECK-NEXT:   Grouped accesses:
; CHECK-NEXT:     Group {{.*}}[[ZERO]]:
; CHECK-NEXT:       (Low: ((2 * %offset) + %a)<nsw> High: (10000 + (2 * %offset) + %a))
; CHECK-NEXT:         Member: {((2 * %offset) + %a)<nsw>,+,2}<nsw><%for.body>
; CHECK-NEXT:     Group {{.*}}[[ONE]]:
; CHECK-NEXT:       (Low: %a High: (10000 + %a))
; CHECK-NEXT:         Member: {%a,+,2}<nw><%for.body>
; CHECK-NEXT:     Group {{.*}}[[TWO]]:
; CHECK-NEXT:       (Low: (20000 + %a) High: (30000 + %a))
; CHECK-NEXT:         Member: {(20000 + %a),+,2}<nw><%for.body>

define void @testi(i16* %a,
                   i64 %offset) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]
  %store_ind = phi i64 [ %offset, %entry ], [ %store_ind_inc, %for.body ]

  %add = add nuw nsw i64 %ind, 1
  %store_ind_inc = add nuw nsw i64 %store_ind, 1

  %arrayidxA1 = getelementptr i16, i16* %a, i64 %ind
  %ind2 = add nuw nsw i64 %ind, 10000
  %arrayidxA2 = getelementptr i16, i16* %a, i64 %ind2

  %loadA1 = load i16, i16* %arrayidxA1, align 2
  %loadA2 = load i16, i16* %arrayidxA2, align 2

  %addres = add i16 %loadA1, %loadA2

  %storeidx = getelementptr inbounds i16, i16* %a, i64 %store_ind
  store i16 %addres, i16* %storeidx, align 2

  %exitcond = icmp eq i64 %add, 5000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
