; REQUIRES: asserts
; RUN: opt -mtriple=s390x-unknown-linux -mcpu=z13 -passes=loop-vectorize \
; RUN:   -debug-only=loop-vectorize,vectorutils -max-interleave-group-factor=64\
; RUN:   -disable-output < %s 2>&1 | FileCheck %s
;
; Check that some cost estimations for interleave groups make sense.

; This loop is loading four i16 values at indices [0, 1, 2, 3], with a stride
; of 4. At VF=4, memory interleaving means loading 4 * 4 * 16 bits = 2 vector
; registers. Each of the 4 vector values must then be constructed from the
; two vector registers using one vperm each, which gives a cost of 2 + 4 = 6.
;
; CHECK: LV: Checking a loop in 'fun0'
; CHECK: Cost of 6 for VF 4: INTERLEAVE-GROUP with factor 4 at %ld0, vp<%next.gep>
; CHECK:   ir<%ld0> = load from index 0
; CHECK:   ir<%ld1> = load from index 1
; CHECK:   ir<%ld2> = load from index 2
; CHECK:   ir<%ld3> = load from index 3

define void @fun0(ptr %ptr, ptr %dst) {
entry:
  br label %for.body

for.body:
  %ivptr = phi ptr [ %ptr.next, %for.body ], [ %ptr, %entry ]
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %inc = add i64 %iv, 4
  %ld0 = load i16, ptr %ivptr
  %ptr1 = getelementptr inbounds i16, ptr %ivptr, i64 1
  %ld1 = load i16, ptr %ptr1
  %ptr2 = getelementptr inbounds i16, ptr %ivptr, i64 2
  %ld2 = load i16, ptr %ptr2
  %ptr3 = getelementptr inbounds i16, ptr %ivptr, i64 3
  %ld3 = load i16, ptr %ptr3
  %a1 = add i16 %ld0, %ld1
  %a2 = add i16 %a1, %ld2
  %a3 = add i16 %a2, %ld3
  %dstptr = getelementptr inbounds i16, ptr %dst, i64 %iv
  store i16 %a3, ptr %dstptr
  %ptr.next = getelementptr inbounds i16, ptr %ivptr, i64 4
  %cmp = icmp eq i64 %inc, 100
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret void
}

; This loop loads one i8 value in a stride of 3. At VF=16, this means loading
; 3 vector registers, and then constructing the vector value with two vperms,
; which gives a cost of 5.
;
; CHECK: LV: Checking a loop in 'fun1'
; CHECK: Cost of 5 for VF 16: INTERLEAVE-GROUP with factor 3 at %ld0, vp<%next.gep>
; CHECK:   ir<%ld0> = load from index 0
define void @fun1(ptr %ptr, ptr %dst) {
entry:
  br label %for.body

for.body:
  %ivptr = phi ptr [ %ptr.next, %for.body ], [ %ptr, %entry ]
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %inc = add i64 %iv, 4
  %ld0 = load i8, ptr %ivptr
  %dstptr = getelementptr inbounds i8, ptr %dst, i64 %iv
  store i8 %ld0, ptr %dstptr
  %ptr.next = getelementptr inbounds i8, ptr %ivptr, i64 3
  %cmp = icmp eq i64 %inc, 100
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret void
}

; This loop is loading 4 i8 values at indexes [0, 1, 2, 3], with a stride of
; 32. At VF=2, this means loading 2 vector registers, and using 4 vperms to
; produce the vector values, which gives a cost of 6.
;
; CHECK: LV: Checking a loop in 'fun2'
; CHECK: Cost of 6 for VF 2: INTERLEAVE-GROUP with factor 32 at %ld0, vp<%next.gep>
; CHECK:   ir<%ld0> = load from index 0
; CHECK:   ir<%ld1> = load from index 1
; CHECK:   ir<%ld2> = load from index 2
; CHECK:   ir<%ld3> = load from index 3
define void @fun2(ptr %ptr, ptr %dst) {
entry:
  br label %for.body

for.body:
  %ivptr = phi ptr [ %ptr.next, %for.body ], [ %ptr, %entry ]
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %inc = add i64 %iv, 4
  %ld0 = load i8, ptr %ivptr
  %ptr1 = getelementptr inbounds i8, ptr %ivptr, i64 1
  %ld1 = load i8, ptr %ptr1
  %ptr2 = getelementptr inbounds i8, ptr %ivptr, i64 2
  %ld2 = load i8, ptr %ptr2
  %ptr3 = getelementptr inbounds i8, ptr %ivptr, i64 3
  %ld3 = load i8, ptr %ptr3
  %a1 = add i8 %ld0, %ld1
  %a2 = add i8 %a1, %ld2
  %a3 = add i8 %a2, %ld3
  %dstptr = getelementptr inbounds i8, ptr %dst, i64 %iv
  store i8 %a3, ptr %dstptr
  %ptr.next = getelementptr inbounds i8, ptr %ivptr, i64 32
  %cmp = icmp eq i64 %inc, 100
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret void
}

; This loop is loading 4 i8 values at indexes [0, 1, 2, 3], with a stride of
; 30. At VF=2, this means loading 3 vector registers, and using 4 vperms to
; produce the vector values, which gives a cost of 7. This is the same loop
; as in fun2, except the stride makes the second iterations values overlap a
; vector register boundary.
;
; CHECK: LV: Checking a loop in 'fun3'
; CHECK: Cost of 7 for VF 2: INTERLEAVE-GROUP with factor 30 at %ld0, vp<%next.gep>
; CHECK:   ir<%ld0> = load from index 0
; CHECK:   ir<%ld1> = load from index 1
; CHECK:   ir<%ld2> = load from index 2
; CHECK:   ir<%ld3> = load from index 3
define void @fun3(ptr %ptr, ptr %dst) {
entry:
  br label %for.body

for.body:
  %ivptr = phi ptr [ %ptr.next, %for.body ], [ %ptr, %entry ]
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %inc = add i64 %iv, 4
  %ld0 = load i8, ptr %ivptr
  %ptr1 = getelementptr inbounds i8, ptr %ivptr, i64 1
  %ld1 = load i8, ptr %ptr1
  %ptr2 = getelementptr inbounds i8, ptr %ivptr, i64 2
  %ld2 = load i8, ptr %ptr2
  %ptr3 = getelementptr inbounds i8, ptr %ivptr, i64 3
  %ld3 = load i8, ptr %ptr3
  %a1 = add i8 %ld0, %ld1
  %a2 = add i8 %a1, %ld2
  %a3 = add i8 %a2, %ld3
  %dstptr = getelementptr inbounds i8, ptr %dst, i64 %iv
  store i8 %a3, ptr %dstptr
  %ptr.next = getelementptr inbounds i8, ptr %ivptr, i64 30
  %cmp = icmp eq i64 %inc, 100
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret void
}
