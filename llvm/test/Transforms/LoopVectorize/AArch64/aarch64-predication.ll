; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -disable-output -debug-only=loop-vectorize 2>&1 | FileCheck %s --check-prefix=COST
; RUN: opt < %s -loop-vectorize -force-vector-width=2 -instcombine -simplifycfg -S | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; This test checks that we correctly compute the scalarized operands for a
; user-specified vectorization factor when interleaving is disabled. We use the
; "optsize" attribute to disable all interleaving calculations.  A cost of 4
; for %tmp4 indicates that we would scalarize it's operand (%tmp3), giving
; %tmp4 a lower scalarization overhead.
;
; COST-LABEL:  predicated_udiv_scalarized_operand
; COST:        LV: Found an estimated cost of 4 for VF 2 For instruction: %tmp4 = udiv i64 %tmp2, %tmp3
;
; CHECK-LABEL: @predicated_udiv_scalarized_operand(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %entry ], [ [[INDEX_NEXT:%.*]], %[[PRED_UDIV_CONTINUE2:.*]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <2 x i64> [ zeroinitializer, %entry ], [ [[TMP17:%.*]], %[[PRED_UDIV_CONTINUE2]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds i64, i64* %a, i64 [[INDEX]]
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i64* [[TMP0]] to <2 x i64>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <2 x i64>, <2 x i64>* [[TMP1]], align 4
; CHECK-NEXT:    [[TMP2:%.*]] = icmp sgt <2 x i64> [[WIDE_LOAD]], zeroinitializer
; CHECK-NEXT:    [[TMP3:%.*]] = extractelement <2 x i1> [[TMP2]], i32 0
; CHECK-NEXT:    br i1 [[TMP3]], label %[[PRED_UDIV_IF:.*]], label %[[PRED_UDIV_CONTINUE:.*]]
; CHECK:       [[PRED_UDIV_IF]]:
; CHECK-NEXT:    [[TMP4:%.*]] = extractelement <2 x i64> [[WIDE_LOAD]], i32 0
; CHECK-NEXT:    [[TMP5:%.*]] = add nsw i64 [[TMP4]], %x
; CHECK-NEXT:    [[TMP6:%.*]] = extractelement <2 x i64> [[WIDE_LOAD]], i32 0
; CHECK-NEXT:    [[TMP7:%.*]] = udiv i64 [[TMP6]], [[TMP5]]
; CHECK-NEXT:    [[TMP8:%.*]] = insertelement <2 x i64> undef, i64 [[TMP7]], i32 0
; CHECK-NEXT:    br label %[[PRED_UDIV_CONTINUE]]
; CHECK:       [[PRED_UDIV_CONTINUE]]:
; CHECK-NEXT:    [[TMP9:%.*]] = phi <2 x i64> [ undef, %vector.body ], [ [[TMP8]], %[[PRED_UDIV_IF]] ]
; CHECK-NEXT:    [[TMP10:%.*]] = extractelement <2 x i1> [[TMP2]], i32 1
; CHECK-NEXT:    br i1 [[TMP10]], label %[[PRED_UDIV_IF1:.*]], label %[[PRED_UDIV_CONTINUE2]]
; CHECK:       [[PRED_UDIV_IF1]]:
; CHECK-NEXT:    [[TMP11:%.*]] = extractelement <2 x i64> [[WIDE_LOAD]], i32 1
; CHECK-NEXT:    [[TMP12:%.*]] = add nsw i64 [[TMP11]], %x
; CHECK-NEXT:    [[TMP13:%.*]] = extractelement <2 x i64> [[WIDE_LOAD]], i32 1
; CHECK-NEXT:    [[TMP14:%.*]] = udiv i64 [[TMP13]], [[TMP12]]
; CHECK-NEXT:    [[TMP15:%.*]] = insertelement <2 x i64> [[TMP9]], i64 [[TMP14]], i32 1
; CHECK-NEXT:    br label %[[PRED_UDIV_CONTINUE2]]
; CHECK:       [[PRED_UDIV_CONTINUE2]]:
; CHECK-NEXT:    [[TMP16:%.*]] = phi <2 x i64> [ [[TMP9]], %[[PRED_UDIV_CONTINUE]] ], [ [[TMP15]], %[[PRED_UDIV_IF1]] ]
; CHECK-NEXT:    [[PREDPHI:%.*]] = select <2 x i1> [[TMP2]], <2 x i64> [[TMP16]], <2 x i64> [[WIDE_LOAD]]
; CHECK-NEXT:    [[TMP17]] = add <2 x i64> [[VEC_PHI]], [[PREDPHI]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 2
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body
;
define i64 @predicated_udiv_scalarized_operand(i64* %a, i64 %x) optsize {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %r = phi i64 [ 0, %entry ], [ %tmp6, %for.inc ]
  %tmp0 = getelementptr inbounds i64, i64* %a, i64 %i
  %tmp2 = load i64, i64* %tmp0, align 4
  %cond0 = icmp sgt i64 %tmp2, 0
  br i1 %cond0, label %if.then, label %for.inc

if.then:
  %tmp3 = add nsw i64 %tmp2, %x
  %tmp4 = udiv i64 %tmp2, %tmp3
  br label %for.inc

for.inc:
  %tmp5 = phi i64 [ %tmp2, %for.body ], [ %tmp4, %if.then]
  %tmp6 = add i64 %r, %tmp5
  %i.next = add nuw nsw i64 %i, 1
  %cond1 = icmp slt i64 %i.next, 100
  br i1 %cond1, label %for.body, label %for.end

for.end:
  %tmp7 = phi i64 [ %tmp6, %for.inc ]
  ret i64 %tmp7
}
