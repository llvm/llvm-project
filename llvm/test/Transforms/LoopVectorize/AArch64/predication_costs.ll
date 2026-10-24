; REQUIRES: asserts
; RUN: opt < %s -force-vector-width=2 -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN:   -enable-early-exit-vectorization-with-side-effects -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; Check predication-related cost calculations, including scalarization overhead
; and block probability scaling. Note that the functionality being tested is
; not specific to AArch64. We specify a target to get actual values for the
; instruction costs.

; CHECK-LABEL: predicated_udiv
;
; This test checks that we correctly compute the cost of the predicated udiv
; instruction. If we assume the block probability is 50%, we compute the cost
; as:
;
; Cost of udiv:
;   (udiv(2) + extractelement(8) + insertelement(4)) / 2 = 7
;
; CHECK: Scalarizing and predicating: %tmp4 = udiv i32 %tmp2, %tmp3
; CHECK: Cost of 7 for VF 2: REPLICATE ir<%tmp4> = udiv ir<%tmp2>, ir<%tmp3> (S->V)
; CHECK: Precomputed costs for VF 2: 4
;
define i32 @predicated_udiv(ptr %a, ptr %b, i1 %c, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %r = phi i32 [ 0, %entry ], [ %tmp6, %for.inc ]
  %tmp0 = getelementptr inbounds i32, ptr %a, i64 %i
  %tmp1 = getelementptr inbounds i32, ptr %b, i64 %i
  %tmp2 = load i32, ptr %tmp0, align 4
  %tmp3 = load i32, ptr %tmp1, align 4
  br i1 %c, label %if.then, label %for.inc

if.then:
  %tmp4 = udiv i32 %tmp2, %tmp3
  br label %for.inc

for.inc:
  %tmp5 = phi i32 [ %tmp3, %for.body ], [ %tmp4, %if.then]
  %tmp6 = add i32 %r, %tmp5
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  %tmp7 = phi i32 [ %tmp6, %for.inc ]
  ret i32 %tmp7
}

; CHECK-LABEL: predicated_store
;
; This test checks that we correctly compute the cost of the predicated store
; instruction. If we assume the block probability is 50%, we compute the cost
; as:
;
; Cost of store:
;   (store(4) + extractelement(4)) / 2 = 4
;
; CHECK: Scalarizing and predicating: store i32 %tmp2, ptr %tmp0, align 4
; CHECK: Cost of 4 for VF 2: profitable to scalarize   store i32 %tmp2, ptr %tmp0, align 4
; CHECK: Precomputed costs for VF 2: 8
;
define void @predicated_store(ptr %a, i1 %c, i32 %x, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %tmp0 = getelementptr inbounds i32, ptr %a, i64 %i
  %tmp1 = load i32, ptr %tmp0, align 4
  %tmp2 = add nsw i32 %tmp1, %x
  br i1 %c, label %if.then, label %for.inc

if.then:
  store i32 %tmp2, ptr %tmp0, align 4
  br label %for.inc

for.inc:
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: predicated_store_phi
;
; Same as predicate_store except we use a pointer PHI to maintain the address
;
; CHECK: Found scalar instruction:   %addr = phi ptr [ %a, %entry ], [ %addr.next, %for.inc ]
; CHECK: Found scalar instruction:   %addr.next = getelementptr inbounds i32, ptr %addr, i64 1
; CHECK: Scalarizing and predicating: store i32 %tmp2, ptr %addr, align 4
; CHECK: Cost of 0 for VF 2: forced scalar   %addr = phi ptr [ %a, %entry ], [ %addr.next, %for.inc ]
; CHECK: Cost of 4 for VF 2: profitable to scalarize   store i32 %tmp2, ptr %addr, align 4
; CHECK: Precomputed costs for VF 2: 8
;
define void @predicated_store_phi(ptr %a, i1 %c, i32 %x, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %addr = phi ptr [ %a, %entry ], [ %addr.next, %for.inc ]
  %tmp1 = load i32, ptr %addr, align 4
  %tmp2 = add nsw i32 %tmp1, %x
  br i1 %c, label %if.then, label %for.inc

if.then:
  store i32 %tmp2, ptr %addr, align 4
  br label %for.inc

for.inc:
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  %addr.next = getelementptr inbounds i32, ptr %addr, i64 1
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: predicated_udiv_scalarized_operand
;
; This test checks that we correctly compute the cost of the predicated udiv
; instruction and the add instruction it uses. The add is scalarized and sunk
; inside the predicated block.  If we assume the block probability is 50%, we
; compute the cost as:
;
; Cost of add:
;   (add(2) + extractelement(4)) / 2 = 3
; Cost of udiv:
;   (udiv(2) + extractelement(4) + insertelement(4)) / 2 = 5
;
; CHECK: Scalarizing: %tmp3 = add nsw i32 %tmp2, %x
; CHECK: Scalarizing and predicating: %tmp4 = udiv i32 %tmp2, %tmp3
; CHECK: Cost of 3 for VF 2: profitable to scalarize   %tmp3 = add nsw i32 %tmp2, %x
; CHECK: Cost of 5 for VF 2: REPLICATE ir<%tmp4> = udiv ir<%tmp2>, ir<%tmp3> (S->V)
; CHECK: Precomputed costs for VF 2: 7
;

define i32 @predicated_udiv_scalarized_operand(ptr %a, i1 %c, i32 %x, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %r = phi i32 [ 0, %entry ], [ %tmp6, %for.inc ]
  %tmp0 = getelementptr inbounds i32, ptr %a, i64 %i
  %tmp2 = load i32, ptr %tmp0, align 4
  br i1 %c, label %if.then, label %for.inc

if.then:
  %tmp3 = add nsw i32 %tmp2, %x
  %tmp4 = udiv i32 %tmp2, %tmp3
  br label %for.inc

for.inc:
  %tmp5 = phi i32 [ %tmp2, %for.body ], [ %tmp4, %if.then]
  %tmp6 = add i32 %r, %tmp5
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  %tmp7 = phi i32 [ %tmp6, %for.inc ]
  ret i32 %tmp7
}

; CHECK-LABEL: predicated_store_scalarized_operand
;
; This test checks that we correctly compute the cost of the predicated store
; instruction and the add instruction it uses. The add is scalarized and sunk
; inside the predicated block.  If we assume the block probability is 50%, we
; compute the cost as:
;
; Cost of add:
;   (add(2) + extractelement(4)) / 2 = 3
; Cost of store:
;   store(4) / 2 = 2
;
; CHECK: Scalarizing and predicating: store i32 %tmp2, ptr %tmp0, align 4
; CHECK: Scalarizing: %tmp2 = add nsw i32 %tmp1, %x
; CHECK: Cost of 2 for VF 2: profitable to scalarize   store i32 %tmp2, ptr %tmp0, align 4
; CHECK: Cost of 3 for VF 2: profitable to scalarize   %tmp2 = add nsw i32 %tmp1, %x
; CHECK: Precomputed costs for VF 2: 9
;
define void @predicated_store_scalarized_operand(ptr %a, i1 %c, i32 %x, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %tmp0 = getelementptr inbounds i32, ptr %a, i64 %i
  %tmp1 = load i32, ptr %tmp0, align 4
  br i1 %c, label %if.then, label %for.inc

if.then:
  %tmp2 = add nsw i32 %tmp1, %x
  store i32 %tmp2, ptr %tmp0, align 4
  br label %for.inc

for.inc:
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: predication_multi_context
;
; This test checks that we correctly compute the cost of multiple predicated
; instructions in the same block. The sdiv, udiv, and store must be scalarized
; and predicated. The sub feeding the store is scalarized and sunk inside the
; store's predicated block. However, the add feeding the sdiv and udiv cannot
; be sunk and is not scalarized. If we assume the block probability is 50%, we
; compute the cost as:
;
; Cost of add:
;   add(1) = 1
; Cost of sdiv:
;   (sdiv(2) + extractelement(8) + insertelement(4)) / 2 = 7
; Cost of udiv:
;   (udiv(2) + extractelement(4) + insertelement(4)) / 2 = 5
; Cost of sub:
;   (sub(2) + extractelement(4)) / 2 = 3
; Cost of store:
;   store(4) / 2 = 2
;
; CHECK-NOT: Scalarizing: %tmp2 = add i32 %tmp1, %x
; CHECK:     Scalarizing and predicating: store i32 %tmp5, ptr %tmp0, align 4
; CHECK-NOT: Scalarizing: %tmp2 = add i32 %tmp1, %x
; CHECK:     Scalarizing and predicating: %tmp3 = sdiv i32 %tmp1, %tmp2
; CHECK:     Scalarizing and predicating: %tmp4 = udiv i32 %tmp3, %tmp2
; CHECK:     Scalarizing: %tmp5 = sub i32 %tmp4, %x
; CHECK: Cost of 2 for VF 2: profitable to scalarize   store i32 %tmp5, ptr %tmp0, align 4
; CHECK: Cost of 3 for VF 2: profitable to scalarize   %tmp5 = sub i32 %tmp4, %x
; CHECK: Cost of 1 for VF 2: WIDEN ir<%tmp2> = add ir<%tmp1>, ir<%x>
; CHECK: Cost of 7 for VF 2: REPLICATE ir<%tmp3> = sdiv ir<%tmp1>, ir<%tmp2>
; CHECK: Cost of 5 for VF 2: REPLICATE ir<%tmp4> = udiv ir<%tmp3>, ir<%tmp2>
; CHECK: Precomputed costs for VF 2: 9
;
define void @predication_multi_context(ptr %a, i1 %c, i32 %x, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %tmp0 = getelementptr inbounds i32, ptr %a, i64 %i
  %tmp1 = load i32, ptr %tmp0, align 4
  br i1 %c, label %if.then, label %for.inc

if.then:
  %tmp2 = add i32 %tmp1, %x
  %tmp3 = sdiv i32 %tmp1, %tmp2
  %tmp4 = udiv i32 %tmp3, %tmp2
  %tmp5 = sub i32 %tmp4, %x
  store i32 %tmp5, ptr %tmp0, align 4
  br label %for.inc

for.inc:
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: predicated_store_trunc_iv
; CHECK: Cost of 2 for VF 2: profitable to scalarize   store i8 %iv.trunc, ptr %gep.dst, align 1
; CHECK-NOT: profitable to scalarize   %iv.trunc = trunc i64 %iv to i8
; CHECK: Precomputed costs for VF 2: 6
; CHECK: Cost of 0 for VF 2: EMIT-SCALAR vp<[[IV_TRUNC:%[0-9]+]]> = trunc vp<[[CAN_IV:%[0-9]+]]> to i8
; CHECK: Cost of 0.5 for VF 2: vp<{{.+}}> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>, vp<{{.+}}>
; CHECK: Cost of 0.5 for VF 2: vp<{{.+}}> = SCALAR-STEPS vp<[[IV_TRUNC]]>, ir<1>, vp<{{.+}}>
; CHECK: Cost of 1 for VF 2: EMIT branch-on-count
; CHECK: Cost of 1 for VF 2: canonical IV increment
; CHECK: Cost for VF 2: 9
;
define void @predicated_store_trunc_iv(ptr %dst, i1 %c, i64 %n) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  br i1 %c, label %then, label %loop.latch

then:
  %iv.trunc = trunc i64 %iv to i8
  %gep.dst = getelementptr i8, ptr %dst, i64 %iv
  store i8 %iv.trunc, ptr %gep.dst
  br label %loop.latch

loop.latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop.header

exit:
  ret void
}

; CHECK-LABEL: early_exit_store_trunc
;
; Check costs for the unpredicated add and trunc feeding the store.
;
; CHECK-NOT: profitable to scalarize
; CHECK: Precomputed costs for VF 2: 4
; CHECK: Cost of 2 for VF 2: REPLICATE ir<%add> = add vp<{{.+}}>, ir<%x>
; CHECK: Cost of 0 for VF 2: REPLICATE ir<%add.trunc> = trunc ir<%add>
; CHECK: Cost of 12 for VF 2: REPLICATE store ir<%add.trunc>, ir<%gep.dst>
; CHECK: Cost for VF 2: 26
;
define void @early_exit_store_trunc(ptr noalias %dst, ptr align 4 dereferenceable(4000) %src, i64 %x) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.src = getelementptr inbounds nuw i32, ptr %src, i64 %iv
  %l = load i32, ptr %gep.src, align 4
  %ee.cond = icmp sgt i32 %l, 500
  br i1 %ee.cond, label %exit, label %loop.latch

loop.latch:
  %add = add i64 %iv, %x
  %add.trunc = trunc i64 %add to i16
  %gep.dst = getelementptr inbounds nuw i16, ptr %dst, i64 %iv
  store i16 %add.trunc, ptr %gep.dst, align 2
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1000
  br i1 %ec, label %exit, label %loop.header

exit:
  ret void
}
