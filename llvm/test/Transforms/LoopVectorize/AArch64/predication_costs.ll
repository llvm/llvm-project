; REQUIRES: asserts
; RUN: opt -force-vplan-cost-model=true < %s -force-vector-width=2 -passes=loop-vectorize -debug-only=loop-vectorize -disable-output 2>&1 | FileCheck %s --check-prefixes=CHECK,VPLAN
; RUN: opt -force-vplan-cost-model=false < %s -force-vector-width=2 -passes=loop-vectorize -debug-only=loop-vectorize -disable-output 2>&1 | FileCheck %s --check-prefixes=CHECK,DEFAULT

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; Check predication-related cost calculations, including scalarization overhead
; and block probability scaling. Note that the functionality being tested is
; not specific to AArch64. We specify a target to get actual values for the
; instruction costs.
;
; The branches of the replicate regions are costed by BRANCH-ON-MASK, which
; extracts each lane of the mask and branches on it:
;   extractelement(2) * 2 + br(0) * 2 = 4

;
; This test checks that we correctly compute the cost of the predicated udiv
; instruction. If we assume the block probability is 50%, we compute the cost
; as:
;
; Cost of udiv:
;   (udiv(2) + extractelement(8) + insertelement(4)) / 2 = 7
;
; CHECK-LABEL: predicated_udiv
; CHECK: Scalarizing and predicating: %tmp4 = udiv i32 %tmp2, %tmp3
; CHECK: Cost of 7 for VF 2: REPLICATE ir<%tmp4> = udiv ir<%tmp2>, ir<%tmp3> (S->V)
; CHECK: Precomputed costs for VF 2: 0
; CHECK: Cost of 4 for VF 2: BRANCH-ON-MASK ir<%c>
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
; CHECK:   Scalarizing and predicating: store i32 %tmp2, ptr %tmp0, align 4
; DEFAULT:    Cost of 4 for VF 2: profitable to scalarize   store i32 %tmp2, ptr %tmp0, align 4
; DEFAULT:    Precomputed costs for VF 2: 4
; VPLAN: Precomputed costs for VF 2: 0
; CHECK:   Cost of 4 for VF 2: BRANCH-ON-MASK ir<%c>
; VPLAN: Cost of 6 for VF 2: REPLICATE store ir<%tmp2>, ir<%tmp0>.1
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
; CHECK:   Found scalar instruction:   %addr = phi ptr [ %a, %entry ], [ %addr.next, %for.inc ]
; CHECK:   Found scalar instruction:   %addr.next = getelementptr inbounds i32, ptr %addr, i64 1
; CHECK:   Scalarizing and predicating: store i32 %tmp2, ptr %addr, align 4
; DEFAULT:    Cost of 4 for VF 2: profitable to scalarize   store i32 %tmp2, ptr %addr, align 4
; DEFAULT:    Precomputed costs for VF 2: 4
; VPLAN: Precomputed costs for VF 2: 0
; CHECK:   Cost of 1 for VF 2: vp<[[DERIVED_IV:%.+]]> = DERIVED-IV ir<0> + vp<{{.+}}> * ir<4>
; CHECK:   Cost of 1 for VF 2: vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[DERIVED_IV]]>, ir<4>, vp<{{.+}}>
; CHECK:   Cost of 0 for VF 2: EMIT vp<%next.gep> = ptradd ir<%a>, vp<[[STEPS]]>
; CHECK:   Cost of 4 for VF 2: BRANCH-ON-MASK ir<%c>
; VPLAN: Cost of 6 for VF 2: REPLICATE store ir<%tmp2>, vp<%next.gep>
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
; CHECK:   Scalarizing: %tmp3 = add nsw i32 %tmp2, %x
; CHECK:   Scalarizing and predicating: %tmp4 = udiv i32 %tmp2, %tmp3
; DEFAULT:    Cost of 3 for VF 2: profitable to scalarize   %tmp3 = add nsw i32 %tmp2, %x
; CHECK:   Cost of 5 for VF 2: REPLICATE ir<%tmp4> = udiv ir<%tmp2>, ir<%tmp3> (S->V)
; DEFAULT:    Precomputed costs for VF 2: 3
; VPLAN: Precomputed costs for VF 2: 0
; CHECK:   Cost of 4 for VF 2: BRANCH-ON-MASK ir<%c>
; VPLAN: Cost of 2 for VF 2: REPLICATE ir<%tmp3> = add nsw ir<%tmp2>, ir<%x>
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
; CHECK:   Scalarizing and predicating: store i32 %tmp2, ptr %tmp0, align 4
; CHECK:   Scalarizing: %tmp2 = add nsw i32 %tmp1, %x
; DEFAULT:    Cost of 2 for VF 2: profitable to scalarize   store i32 %tmp2, ptr %tmp0, align 4
; DEFAULT:    Cost of 3 for VF 2: profitable to scalarize   %tmp2 = add nsw i32 %tmp1, %x
; DEFAULT:    Precomputed costs for VF 2: 5
; VPLAN: Precomputed costs for VF 2: 0
; CHECK:   Cost of 4 for VF 2: BRANCH-ON-MASK ir<%c>
; VPLAN: Cost of 2 for VF 2: REPLICATE ir<%tmp2> = add nsw ir<%tmp1>, ir<%x>
; VPLAN: Cost of 6 for VF 2: REPLICATE store ir<%tmp2>, ir<%tmp0>.1
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
; DEFAULT:      Cost of 2 for VF 2: profitable to scalarize   store i32 %tmp5, ptr %tmp0, align 4
; DEFAULT:      Cost of 3 for VF 2: profitable to scalarize   %tmp5 = sub i32 %tmp4, %x
; CHECK:     Cost of 1 for VF 2: WIDEN ir<%tmp2> = add ir<%tmp1>, ir<%x>
; CHECK:     Cost of 7 for VF 2: REPLICATE ir<%tmp3> = sdiv ir<%tmp1>, ir<%tmp2>
; CHECK:     Cost of 5 for VF 2: REPLICATE ir<%tmp4> = udiv ir<%tmp3>, ir<%tmp2>
; DEFAULT:      Precomputed costs for VF 2: 5
; VPLAN:   Precomputed costs for VF 2: 0
; CHECK:     Cost of 4 for VF 2: BRANCH-ON-MASK ir<%c>
; VPLAN:   Cost of 2 for VF 2: REPLICATE ir<%tmp5> = sub ir<%tmp4>, ir<%x>
; VPLAN:   Cost of 6 for VF 2: REPLICATE store ir<%tmp5>, ir<%tmp0>.1
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
; DEFAULT: Cost of 2 for VF 2: profitable to scalarize   store i8 %iv.trunc, ptr %gep.dst, align 1
; DEFAULT-NOT: profitable to scalarize   %iv.trunc = trunc i64 %iv to i8
; DEFAULT: Precomputed costs for VF 2: 2
; VPLAN: Precomputed costs for VF 2: 0
; CHECK:   Cost of 0 for VF 2: EMIT-SCALAR vp<[[IV_TRUNC:%[0-9]+]]> = trunc vp<[[CAN_IV:%[0-9]+]]> to i8
; CHECK:   Cost of 4 for VF 2: BRANCH-ON-MASK ir<%c>
; CHECK:   Cost of 0.5 for VF 2: vp<{{.+}}> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>, vp<{{.+}}>
; CHECK:   Cost of 0.5 for VF 2: vp<{{.+}}> = SCALAR-STEPS vp<[[IV_TRUNC]]>, ir<1>, vp<{{.+}}>
; VPLAN: Cost of 6 for VF 2: REPLICATE store vp<%6>, ir<%gep.dst>
; CHECK:   Cost of 1 for VF 2: EMIT branch-on-count
; CHECK:   Cost of 1 for VF 2: canonical IV increment
; DEFAULT:    Cost for VF 2: 9
; VPLAN: Cost for VF 2: 13
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

; CHECK-LABEL: predicated_udiv_operand_from_other_region
;
; The udiv in the nested predicated block uses the udiv from the outer
; predicated block, which is in a different replicate region. If we assume the
; block probabilities are 50% and 25%, we compute the costs as:
;
; Cost of %tmp2:
;   (udiv(2) + extractelement(4) + insertelement(4)) / 2 = 5
; Cost of %tmp3, including extracting the lanes of %tmp2, which is packed for
; the phi:
;   (udiv(2) + extractelement(4) + insertelement(4)) / 4 = 2.5
;
; CHECK:   Scalarizing and predicating: %tmp2 = udiv i32 %tmp1, %x
; CHECK:   Scalarizing and predicating: %tmp3 = udiv i32 %tmp2, %y
; CHECK:   Cost of 5 for VF 2: REPLICATE ir<%tmp2> = udiv ir<%tmp1>, ir<%x> (S->V)
; DEFAULT:    Cost of 2.5 for VF 2: REPLICATE ir<%tmp3> = udiv vp<{{.+}}>, ir<%y> (S->V)
; VPLAN: Cost of 1.5 for VF 2: REPLICATE ir<%tmp3> = udiv vp<{{.+}}>, ir<%y> (S->V)
;
define void @predicated_udiv_operand_from_other_region(ptr %a, i1 %c, i1 %c2, i32 %x, i32 %y, i64 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop.latch ]
  %tmp0 = getelementptr inbounds i32, ptr %a, i64 %i
  %tmp1 = load i32, ptr %tmp0, align 4
  br i1 %c, label %if.then, label %loop.latch

if.then:
  %tmp2 = udiv i32 %tmp1, %x
  br i1 %c2, label %if.then2, label %loop.latch

if.then2:
  %tmp3 = udiv i32 %tmp2, %y
  br label %loop.latch

loop.latch:
  %tmp4 = phi i32 [ %tmp1, %loop.header ], [ %tmp2, %if.then ], [ %tmp3, %if.then2 ]
  store i32 %tmp4, ptr %tmp0, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %loop.header, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: predicated_udiv_operand_from_other_region_no_vector_user
;
; Same as predicated_udiv_operand_from_other_region, but %tmp2 only has scalar
; users and is not packed. The costs include inserting and extracting the lanes
; of %tmp2:
;
; Cost of %tmp2:
;   (udiv(2) + extractelement(4) + insertelement(4)) / 2 = 5
; Cost of %tmp3, including extracting the lanes of %tmp2:
;   (udiv(2) + extractelement(4) + insertelement(4)) / 4 = 2.5
;
; CHECK:   Scalarizing and predicating: %tmp2 = udiv i32 %tmp1, %x
; CHECK:   Scalarizing and predicating: %tmp3 = udiv i32 %tmp2, %y
; CHECK:   Cost of 5 for VF 2: REPLICATE ir<%tmp2> = udiv ir<%tmp1>, ir<%x>{{$}}
; DEFAULT:    Cost of 2.5 for VF 2: REPLICATE ir<%tmp3> = udiv vp<{{.+}}>, ir<%y> (S->V)
; VPLAN: Cost of 1.5 for VF 2: REPLICATE ir<%tmp3> = udiv vp<{{.+}}>, ir<%y> (S->V)
;
define void @predicated_udiv_operand_from_other_region_no_vector_user(ptr %a, i1 %c, i1 %c2, i32 %x, i32 %y, i64 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop.latch ]
  %tmp0 = getelementptr inbounds i32, ptr %a, i64 %i
  %tmp1 = load i32, ptr %tmp0, align 4
  br i1 %c, label %if.then, label %loop.latch

if.then:
  %tmp2 = udiv i32 %tmp1, %x
  br i1 %c2, label %if.then2, label %loop.latch

if.then2:
  %tmp3 = udiv i32 %tmp2, %y
  br label %loop.latch

loop.latch:
  %tmp4 = phi i32 [ %tmp1, %loop.header ], [ %tmp1, %if.then ], [ %tmp3, %if.then2 ]
  store i32 %tmp4, ptr %tmp0, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %loop.header, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: predicated_load_operand_from_other_region
;
; Same as predicated_udiv_operand_from_other_region, but %tmp2 is a predicated
; load. The cost of %tmp3 does not include extracting the lanes of %tmp2, which
; is packed for the phi:
;
; Cost of %tmp3:
;   (udiv(2) + insertelement(4)) / 4 = 1.5
;
; CHECK: Scalarizing and predicating: %tmp2 = load i32, ptr %gep.b, align 4
; CHECK: Scalarizing and predicating: %tmp3 = udiv i32 %tmp2, %y
; CHECK: Cost of 1.5 for VF 2: REPLICATE ir<%tmp3> = udiv vp<{{.+}}>, ir<%y> (S->V)
;
define void @predicated_load_operand_from_other_region(ptr %a, ptr %b, i1 %c, i1 %c2, i32 %y, i64 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop.latch ]
  %tmp0 = getelementptr inbounds i32, ptr %a, i64 %i
  %tmp1 = load i32, ptr %tmp0, align 4
  br i1 %c, label %if.then, label %loop.latch

if.then:
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %i
  %tmp2 = load i32, ptr %gep.b, align 4
  br i1 %c2, label %if.then2, label %loop.latch

if.then2:
  %tmp3 = udiv i32 %tmp2, %y
  br label %loop.latch

loop.latch:
  %tmp4 = phi i32 [ %tmp1, %loop.header ], [ %tmp2, %if.then ], [ %tmp3, %if.then2 ]
  store i32 %tmp4, ptr %tmp0, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %loop.header, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: predicated_udiv_used_by_call
;
; The udiv is used by a speculatable call, which is replicated outside the
; replicate region, and by a widened add, so %tmp2 is packed. The cost of the
; call does not include extracting the lanes of %tmp2. If we assume the block
; probability is 50%, we compute the costs as:
;
; Cost of %tmp2:
;   (udiv(2) + extractelement(4) + insertelement(4)) / 2 = 5
; Cost of %tmp3:
;   call(20) + insertelement(4) = 24
;
; CHECK: Scalarizing: %tmp3 = call i32 @foo(i32 %tmp2)
; CHECK: Scalarizing and predicating: %tmp2 = udiv i32 %tmp1, %x
; CHECK: Cost of 5 for VF 2: REPLICATE ir<%tmp2> = udiv ir<%tmp1>, ir<%x> (S->V)
; CHECK: Cost of 24 for VF 2: REPLICATE ir<%tmp3> = call @foo(vp<{{.+}}>)
;
define void @predicated_udiv_used_by_call(ptr %a, i1 %c, i32 %x, i64 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop.latch ]
  %tmp0 = getelementptr inbounds i32, ptr %a, i64 %i
  %tmp1 = load i32, ptr %tmp0, align 4
  br i1 %c, label %if.then, label %loop.latch

if.then:
  %tmp2 = udiv i32 %tmp1, %x
  %tmp3 = call i32 @foo(i32 %tmp2) #0
  %tmp4 = add i32 %tmp3, %tmp2
  br label %loop.latch

loop.latch:
  %tmp5 = phi i32 [ %tmp1, %loop.header ], [ %tmp4, %if.then ]
  store i32 %tmp5, ptr %tmp0, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %loop.header, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: predicated_udiv_used_by_call_no_vector_user
;
; Same as predicated_udiv_used_by_call, but %tmp2 is only used by the call and
; is not packed. We compute the costs as:
;
; Cost of %tmp2:
;   (udiv(2) + extractelement(4) + insertelement(4)) / 2 = 5
; Cost of %tmp3:
;   call(20) + insertelement(4) = 24
;
; CHECK: Scalarizing: %tmp3 = call i32 @foo(i32 %tmp2)
; CHECK: Scalarizing and predicating: %tmp2 = udiv i32 %tmp1, %x
; CHECK: Cost of 5 for VF 2: REPLICATE ir<%tmp2> = udiv ir<%tmp1>, ir<%x>{{$}}
; CHECK: Cost of 24 for VF 2: REPLICATE ir<%tmp3> = call @foo(vp<{{.+}}>)
;
define void @predicated_udiv_used_by_call_no_vector_user(ptr %a, i1 %c, i32 %x, i64 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop.latch ]
  %tmp0 = getelementptr inbounds i32, ptr %a, i64 %i
  %tmp1 = load i32, ptr %tmp0, align 4
  br i1 %c, label %if.then, label %loop.latch

if.then:
  %tmp2 = udiv i32 %tmp1, %x
  %tmp3 = call i32 @foo(i32 %tmp2) #0
  br label %loop.latch

loop.latch:
  %tmp4 = phi i32 [ %tmp1, %loop.header ], [ %tmp3, %if.then ]
  store i32 %tmp4, ptr %tmp0, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %loop.header, label %for.end

for.end:
  ret void
}

declare i32 @foo(i32) speculatable willreturn nounwind memory(none)
declare <4 x i32> @foo_vec(<4 x i32>)

; The N4 variant makes the calls legal to vectorize; VF 2 replicates them.
attributes #0 = { "vector-function-abi-variant"="_ZGVnN4v_foo(foo_vec)" }
