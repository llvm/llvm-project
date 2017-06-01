; RUN: opt < %s -loop-vectorize -simplifycfg -S | FileCheck %s
; RUN: opt < %s -force-vector-width=2 -loop-vectorize -simplifycfg -S | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; CHECK-LABEL: predicated_udiv_scalarized_operand
;
; This test checks that we correctly compute the scalarized operands for a
; user-specified vectorization factor when interleaving is disabled. We use the
; "optsize" attribute to disable all interleaving calculations.
;
; CHECK: vector.body:
; CHECK:   %wide.load = load <2 x i64>, <2 x i64>* {{.*}}, align 4
; CHECK:   br i1 {{.*}}, label %[[IF0:.+]], label %[[CONT0:.+]]
; CHECK: [[IF0]]:
; CHECK:   %[[T00:.+]] = extractelement <2 x i64> %wide.load, i32 0
; CHECK:   %[[T01:.+]] = extractelement <2 x i64> %wide.load, i32 0
; CHECK:   %[[T02:.+]] = add nsw i64 %[[T01]], %x
; CHECK:   %[[T03:.+]] = udiv i64 %[[T00]], %[[T02]]
; CHECK:   %[[T04:.+]] = insertelement <2 x i64> undef, i64 %[[T03]], i32 0
; CHECK:   br label %[[CONT0]]
; CHECK: [[CONT0]]:
; CHECK:   %[[T05:.+]] = phi <2 x i64> [ undef, %vector.body ], [ %[[T04]], %[[IF0]] ]
; CHECK:   br i1 {{.*}}, label %[[IF1:.+]], label %[[CONT1:.+]]
; CHECK: [[IF1]]:
; CHECK:   %[[T06:.+]] = extractelement <2 x i64> %wide.load, i32 1
; CHECK:   %[[T07:.+]] = extractelement <2 x i64> %wide.load, i32 1
; CHECK:   %[[T08:.+]] = add nsw i64 %[[T07]], %x
; CHECK:   %[[T09:.+]] = udiv i64 %[[T06]], %[[T08]]
; CHECK:   %[[T10:.+]] = insertelement <2 x i64> %[[T05]], i64 %[[T09]], i32 1
; CHECK:   br label %[[CONT1]]
; CHECK: [[CONT1]]:
; CHECK:   phi <2 x i64> [ %[[T05]], %[[CONT0]] ], [ %[[T10]], %[[IF1]] ]
; CHECK:   br i1 {{.*}}, label %middle.block, label %vector.body

define i64 @predicated_udiv_scalarized_operand(i64* %a, i1 %c, i64 %x) optsize {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %r = phi i64 [ 0, %entry ], [ %tmp6, %for.inc ]
  %tmp0 = getelementptr inbounds i64, i64* %a, i64 %i
  %tmp2 = load i64, i64* %tmp0, align 4
  br i1 %c, label %if.then, label %for.inc

if.then:
  %tmp3 = add nsw i64 %tmp2, %x
  %tmp4 = udiv i64 %tmp2, %tmp3
  br label %for.inc

for.inc:
  %tmp5 = phi i64 [ %tmp2, %for.body ], [ %tmp4, %if.then]
  %tmp6 = add i64 %r, %tmp5
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, 100
  br i1 %cond, label %for.body, label %for.end

for.end:
  %tmp7 = phi i64 [ %tmp6, %for.inc ]
  ret i64 %tmp7
}
