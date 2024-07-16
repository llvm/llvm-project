; RUN: opt %s -passes=loop-vectorize -force-vector-interleave=3 -force-vector-width=4 -S | FileCheck --check-prefix=UF3 %s
; RUN: opt %s -passes=loop-vectorize -force-vector-interleave=5 -force-vector-width=4 -S | FileCheck --check-prefix=UF5 %s

define i32 @reduction_sum(i64 %n, ptr noalias nocapture %A) {
; UF3-LABEL: vector.body:
; UF3-NEXT:   [[IV:%.+]] = phi i64 [ 0, %vector.ph ], [ [[IV_NEXT:%.+]], %vector.body ]
; UF3-NEXT:   [[SUM0:%.+]] = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ [[SUM0_NEXT:%.+]], %vector.body ]
; UF3-NEXT:   [[SUM1:%.+]] = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ [[SUM1_NEXT:%.+]], %vector.body ]
; UF3-NEXT:   [[SUM2:%.+]] = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ [[SUM2_NEXT:%.+]],  %vector.body ]
; UF3-NEXT:   [[IV0:%.+]] = add i64 [[IV]], 0
; UF3-NEXT:   [[IV1:%.+]] = add i64 [[IV]], 4
; UF3-NEXT:   [[IV2:%.+]] = add i64 [[IV]], 8
; UF3-NEXT:   [[GEP0:%.+]] = getelementptr inbounds i32, ptr %A, i64 [[IV0]]
; UF3-NEXT:   [[GEP1:%.+]] = getelementptr inbounds i32, ptr %A, i64 [[IV1]]
; UF3-NEXT:   [[GEP2:%.+]] = getelementptr inbounds i32, ptr %A, i64 [[IV2]]
; UF3-NEXT:   [[L_GEP0:%.+]] = getelementptr inbounds i32, ptr [[GEP0]], i32 0
; UF3-NEXT:   [[L_GEP1:%.+]] = getelementptr inbounds i32, ptr [[GEP0]], i32 4
; UF3-NEXT:   [[L_GEP2:%.+]] = getelementptr inbounds i32, ptr [[GEP0]], i32 8
; UF3-NEXT:   [[L0:%.+]] = load <4 x i32>, ptr [[L_GEP0]], align 4
; UF3-NEXT:   [[L1:%.+]] = load <4 x i32>, ptr [[L_GEP1]], align 4
; UF3-NEXT:   [[L2:%.+]] = load <4 x i32>, ptr [[L_GEP2]], align 4
; UF3-NEXT:   [[SUM0_NEXT]] = add <4 x i32> [[SUM0]], [[L0]]
; UF3-NEXT:   [[SUM1_NEXT]] = add <4 x i32> [[SUM1]], [[L1]]
; UF3-NEXT:   [[SUM2_NEXT]] = add <4 x i32> [[SUM2]], [[L2]]
; UF3-NEXT:   [[IV_NEXT]] = add nuw i64 [[IV]], 12
; UF3-NEXT:   [[EC:%.+]] = icmp eq i64 [[IV_NEXT]], %n.vec
; UF3-NEXT:   br i1 [[EC]], label %middle.block, label %vector.body
;
; UF3-LABEL: middle.block:
; UF3-NEXT:   [[RDX0:%.+]] = add <4 x i32> [[SUM1_NEXT]], [[SUM0_NEXT]]
; UF3-NEXT:   [[RDX1:%.+]] = add <4 x i32> [[SUM2_NEXT]], [[RDX0]]
; UF3-NEXT:   call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[RDX1]])
;

; UF5-LABEL: vector.body:
; UF5-NEXT:   [[IV:%.+]] = phi i64 [ 0, %vector.ph ], [ [[IV_NEXT:%.+]], %vector.body ]
; UF5-NEXT:   [[SUM0:%.+]] = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ [[SUM0_NEXT:%.+]], %vector.body ]
; UF5-NEXT:   [[SUM1:%.+]] = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ [[SUM1_NEXT:%.+]], %vector.body ]
; UF5-NEXT:   [[SUM2:%.+]] = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ [[SUM2_NEXT:%.+]],  %vector.body ]
; UF5-NEXT:   [[SUM3:%.+]] = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ [[SUM3_NEXT:%.+]], %vector.body ]
; UF5-NEXT:   [[SUM4:%.+]] = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ [[SUM4_NEXT:%.+]], %vector.body ]
; UF5-NEXT:   [[IV0:%.+]] = add i64 [[IV]], 0
; UF5-NEXT:   [[IV1:%.+]] = add i64 [[IV]], 4
; UF5-NEXT:   [[IV2:%.+]] = add i64 [[IV]], 8
; UF5-NEXT:   [[IV3:%.+]] = add i64 [[IV]], 12
; UF5-NEXT:   [[IV4:%.+]] = add i64 [[IV]], 16
; UF5-NEXT:   [[GEP0:%.+]] = getelementptr inbounds i32, ptr %A, i64 [[IV0]]
; UF5-NEXT:   [[GEP1:%.+]] = getelementptr inbounds i32, ptr %A, i64 [[IV1]]
; UF5-NEXT:   [[GEP2:%.+]] = getelementptr inbounds i32, ptr %A, i64 [[IV2]]
; UF5-NEXT:   [[GEP3:%.+]] = getelementptr inbounds i32, ptr %A, i64 [[IV3]]
; UF5-NEXT:   [[GEP4:%.+]] = getelementptr inbounds i32, ptr %A, i64 [[IV4]]
; UF5-NEXT:   [[L_GEP0:%.+]] = getelementptr inbounds i32, ptr [[GEP0]], i32 0
; UF5-NEXT:   [[L_GEP1:%.+]] = getelementptr inbounds i32, ptr [[GEP0]], i32 4
; UF5-NEXT:   [[L_GEP2:%.+]] = getelementptr inbounds i32, ptr [[GEP0]], i32 8
; UF5-NEXT:   [[L_GEP3:%.+]] = getelementptr inbounds i32, ptr [[GEP0]], i32 12
; UF5-NEXT:   [[L_GEP4:%.+]] = getelementptr inbounds i32, ptr [[GEP0]], i32 16
; UF5-NEXT:   [[L0:%.+]] = load <4 x i32>, ptr [[L_GEP0]], align 4
; UF5-NEXT:   [[L1:%.+]] = load <4 x i32>, ptr [[L_GEP1]], align 4
; UF5-NEXT:   [[L2:%.+]] = load <4 x i32>, ptr [[L_GEP2]], align 4
; UF5-NEXT:   [[L3:%.+]] = load <4 x i32>, ptr [[L_GEP3]], align 4
; UF5-NEXT:   [[L4:%.+]] = load <4 x i32>, ptr [[L_GEP4]], align 4
; UF5-NEXT:   [[SUM0_NEXT]] = add <4 x i32> [[SUM0]], [[L0]]
; UF5-NEXT:   [[SUM1_NEXT]] = add <4 x i32> [[SUM1]], [[L1]]
; UF5-NEXT:   [[SUM2_NEXT]] = add <4 x i32> [[SUM2]], [[L2]]
; UF5-NEXT:   [[SUM3_NEXT]] = add <4 x i32> [[SUM3]], [[L3]]
; UF5-NEXT:   [[SUM4_NEXT]] = add <4 x i32> [[SUM4]], [[L4]]
; UF5-NEXT:   [[IV_NEXT]] = add nuw i64 [[IV]], 20
; UF5-NEXT:   [[EC:%.+]] = icmp eq i64 [[IV_NEXT]], %n.vec
; UF5-NEXT:   br i1 [[EC]], label %middle.block, label %vector.body
;
; UF5-LABEL: middle.block:
; UF5-NEXT:   [[RDX0:%.+]] = add <4 x i32> [[SUM1_NEXT]], [[SUM0_NEXT]]
; UF5-NEXT:   [[RDX1:%.+]] = add <4 x i32> [[SUM2_NEXT]], [[RDX0]]
; UF5-NEXT:   [[RDX2:%.+]] = add <4 x i32> [[SUM3_NEXT]], [[RDX1]]
; UF5-NEXT:   [[RDX3:%.+]] = add <4 x i32> [[SUM4_NEXT]], [[RDX2]]
; UF5-NEXT:   call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[RDX3]])
;

entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %sum.02 = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  %gep.A = getelementptr inbounds i32, ptr %A, i64 %iv
  %lv.A = load i32, ptr %gep.A, align 4
  %sum.next = add i32 %sum.02, %lv.A
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  %sum.0.lcssa = phi i32 [ %sum.next, %loop ]
  ret i32 %sum.0.lcssa
}
