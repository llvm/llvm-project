; REQUIRES: asserts
; RUN: opt < %s -force-vector-width=2 -passes=loop-vectorize -debug-only=loop-vectorize -disable-output 2>&1 | FileCheck %s

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
;   (udiv(2) + extractelement(6) + insertelement(3)) / 2 = 5
;
; CHECK: Scalarizing and predicating: %tmp4 = udiv i32 %tmp2, %tmp3
; CHECK: Found an estimated cost of 5 for VF 2 For instruction: %tmp4 = udiv i32 %tmp2, %tmp3
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
;   (store(4) + extractelement(3)) / 2 = 3
;
; CHECK: Scalarizing and predicating: store i32 %tmp2, ptr %tmp0, align 4
; CHECK: Found an estimated cost of 3 for VF 2 For instruction: store i32 %tmp2, ptr %tmp0, align 4
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
; CHECK: Found an estimated cost of 0 for VF 2 For instruction:   %addr = phi ptr [ %a, %entry ], [ %addr.next, %for.inc ]
; CHECK: Found an estimated cost of 3 for VF 2 For instruction: store i32 %tmp2, ptr %addr, align 4
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
;   (add(2) + extractelement(3)) / 2 = 2
; Cost of udiv:
;   (udiv(2) + extractelement(3) + insertelement(3)) / 2 = 4
;
; CHECK: Scalarizing: %tmp3 = add nsw i32 %tmp2, %x
; CHECK: Scalarizing and predicating: %tmp4 = udiv i32 %tmp2, %tmp3
; CHECK: Found an estimated cost of 2 for VF 2 For instruction: %tmp3 = add nsw i32 %tmp2, %x
; CHECK: Found an estimated cost of 4 for VF 2 For instruction: %tmp4 = udiv i32 %tmp2, %tmp3
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
;   (add(2) + extractelement(3)) / 2 = 2
; Cost of store:
;   store(4) / 2 = 2
;
; CHECK: Scalarizing: %tmp2 = add nsw i32 %tmp1, %x
; CHECK: Scalarizing and predicating: store i32 %tmp2, ptr %tmp0, align 4
; CHECK: Found an estimated cost of 2 for VF 2 For instruction: %tmp2 = add nsw i32 %tmp1, %x
; CHECK: Found an estimated cost of 2 for VF 2 For instruction: store i32 %tmp2, ptr %tmp0, align 4
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
;   (sdiv(2) + extractelement(6) + insertelement(3)) / 2 = 5
; Cost of udiv:
;   (udiv(2) + extractelement(6) + insertelement(3)) / 2 = 5
; Cost of sub:
;   (sub(2) + extractelement(3)) / 2 = 2
; Cost of store:
;   store(4) / 2 = 2
;
; CHECK-NOT: Scalarizing: %tmp2 = add i32 %tmp1, %x
; CHECK:     Scalarizing and predicating: %tmp3 = sdiv i32 %tmp1, %tmp2
; CHECK:     Scalarizing and predicating: %tmp4 = udiv i32 %tmp3, %tmp2
; CHECK:     Scalarizing: %tmp5 = sub i32 %tmp4, %x
; CHECK:     Scalarizing and predicating: store i32 %tmp5, ptr %tmp0, align 4
; CHECK:     Found an estimated cost of 1 for VF 2 For instruction: %tmp2 = add i32 %tmp1, %x
; CHECK:     Found an estimated cost of 5 for VF 2 For instruction: %tmp3 = sdiv i32 %tmp1, %tmp2
; CHECK:     Found an estimated cost of 5 for VF 2 For instruction: %tmp4 = udiv i32 %tmp3, %tmp2
; CHECK:     Found an estimated cost of 2 for VF 2 For instruction: %tmp5 = sub i32 %tmp4, %x
; CHECK:     Found an estimated cost of 2 for VF 2 For instruction: store i32 %tmp5, ptr %tmp0, align 4
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
