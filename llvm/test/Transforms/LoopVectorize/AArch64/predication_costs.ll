; REQUIRES: asserts
; RUN: opt < %s -force-vector-width=2 -enable-cond-stores-vec -loop-vectorize -debug-only=loop-vectorize -disable-output 2>&1 | FileCheck %s

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
; CHECK: Found an estimated cost of 5 for VF 2 For instruction: %tmp4 = udiv i32 %tmp2, %tmp3
; CHECK: Scalarizing and predicating: %tmp4 = udiv i32 %tmp2, %tmp3
;
define i32 @predicated_udiv(i32* %a, i32* %b, i1 %c, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %r = phi i32 [ 0, %entry ], [ %tmp6, %for.inc ]
  %tmp0 = getelementptr inbounds i32, i32* %a, i64 %i
  %tmp1 = getelementptr inbounds i32, i32* %b, i64 %i
  %tmp2 = load i32, i32* %tmp0, align 4
  %tmp3 = load i32, i32* %tmp1, align 4
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
;   (store(4) + extractelement(6)) / 2 = 5
;
; CHECK: Found an estimated cost of 5 for VF 2 For instruction: store i32 %tmp2, i32* %tmp0, align 4
; CHECK: Scalarizing and predicating: store i32 %tmp2, i32* %tmp0, align 4
;
define void @predicated_store(i32* %a, i1 %c, i32 %x, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %tmp0 = getelementptr inbounds i32, i32* %a, i64 %i
  %tmp1 = load i32, i32* %tmp0, align 4
  %tmp2 = add nsw i32 %tmp1, %x
  br i1 %c, label %if.then, label %for.inc

if.then:
  store i32 %tmp2, i32* %tmp0, align 4
  br label %for.inc

for.inc:
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}
