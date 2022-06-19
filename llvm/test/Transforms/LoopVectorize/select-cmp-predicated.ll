; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=2 -S < %s | FileCheck %s --check-prefix=CHECK-VF2IC1
; RUN: opt -passes=loop-vectorize -force-vector-interleave=2 -force-vector-width=1 -S < %s | FileCheck %s --check-prefix=CHECK-VF1IC2

define i32 @pred_select_const_i32_from_icmp(i32* noalias nocapture readonly %src1, i32* noalias nocapture readonly %src2, i64 %n) {
; CHECK-VF2IC1-LABEL: @pred_select_const_i32_from_icmp(
; CHECK-VF2IC1:       vector.body:
; CHECK-VF2IC1:         [[VEC_PHI:%.*]] = phi <2 x i32> [ zeroinitializer, %vector.ph ], [ [[PREDPHI:%.*]], %pred.load.continue2 ]
; CHECK-VF2IC1:         [[WIDE_LOAD:%.*]] = load <2 x i32>, <2 x i32>* {{%.*}}, align 4
; CHECK-VF2IC1-NEXT:    [[TMP4:%.*]] = icmp sgt <2 x i32> [[WIDE_LOAD]], <i32 35, i32 35>
; CHECK-VF2IC1-NEXT:    [[TMP5:%.*]] = extractelement <2 x i1> [[TMP4]], i32 0
; CHECK-VF2IC1-NEXT:    br i1 [[TMP5]], label %pred.load.if, label %pred.load.continue
; CHECK-VF2IC1:       pred.load.if:
; CHECK-VF2IC1-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, i32* [[SRC2:%.*]], i64 {{%.*}}
; CHECK-VF2IC1-NEXT:    [[TMP7:%.*]] = load i32, i32* [[TMP6]], align 4
; CHECK-VF2IC1-NEXT:    [[TMP8:%.*]] = insertelement <2 x i32> poison, i32 [[TMP7]], i32 0
; CHECK-VF2IC1-NEXT:    br label %pred.load.continue
; CHECK-VF2IC1:       pred.load.continue:
; CHECK-VF2IC1-NEXT:    [[TMP9:%.*]] = phi <2 x i32> [ poison, %vector.body ], [ [[TMP8]], %pred.load.if ]
; CHECK-VF2IC1-NEXT:    [[TMP10:%.*]] = extractelement <2 x i1> [[TMP4]], i32 1
; CHECK-VF2IC1-NEXT:    br i1 [[TMP10]], label %pred.load.if1, label %pred.load.continue2
; CHECK-VF2IC1:       pred.load.if1:
; CHECK-VF2IC1:         [[TMP12:%.*]] = getelementptr inbounds i32, i32* [[SRC2]], i64 {{%.*}}
; CHECK-VF2IC1-NEXT:    [[TMP13:%.*]] = load i32, i32* [[TMP12]], align 4
; CHECK-VF2IC1-NEXT:    [[TMP14:%.*]] = insertelement <2 x i32> [[TMP9]], i32 [[TMP13]], i32 1
; CHECK-VF2IC1-NEXT:    br label %pred.load.continue2
; CHECK-VF2IC1:       pred.load.continue2:
; CHECK-VF2IC1-NEXT:    [[TMP15:%.*]] = phi <2 x i32> [ [[TMP9]], %pred.load.continue ], [ [[TMP14]], %pred.load.if1 ]
; CHECK-VF2IC1-NEXT:    [[TMP16:%.*]] = icmp eq <2 x i32> [[TMP15]], <i32 2, i32 2>
; CHECK-VF2IC1-NEXT:    [[TMP17:%.*]] = select <2 x i1> [[TMP16]], <2 x i32> <i32 1, i32 1>, <2 x i32> [[VEC_PHI]]
; CHECK-VF2IC1-NEXT:    [[TMP18:%.*]] = xor <2 x i1> [[TMP4]], <i1 true, i1 true>
; CHECK-VF2IC1-NEXT:    [[PREDPHI]] = select <2 x i1> [[TMP4]], <2 x i32> [[TMP17]], <2 x i32> [[VEC_PHI]]
; CHECK-VF2IC1:         br i1 {{%.*}}, label %middle.block, label %vector.body
; CHECK-VF2IC1:       middle.block:
; CHECK-VF2IC1-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne <2 x i32> [[PREDPHI]], zeroinitializer
; CHECK-VF2IC1-NEXT:    [[TMP20:%.*]] = call i1 @llvm.vector.reduce.or.v2i1(<2 x i1> [[RDX_SELECT_CMP]])
; CHECK-VF2IC1-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[TMP20]], i32 1, i32 0
; CHECK-VF2IC1:       scalar.ph:
; CHECK-VF2IC1:         [[BC_RESUME_VAL:%.*]] = phi i64 [ {{%.*}}, %middle.block ], [ 0, %entry ]
; CHECK-VF2IC1-NEXT:    [[BC_MERGE_RDX:%.*]] = phi i32 [ 0, %entry ], [ [[RDX_SELECT]], %middle.block ]
; CHECK-VF2IC1-NEXT:    br label %for.body
; CHECK-VF2IC1:       for.body:
; CHECK-VF2IC1:         [[R_012:%.*]] = phi i32 [ [[R_1:%.*]], %for.inc ], [ [[BC_MERGE_RDX]], %scalar.ph ]
; CHECK-VF2IC1:         [[TMP21:%.*]] = load i32, i32* {{%.*}}, align 4
; CHECK-VF2IC1-NEXT:    [[CMP1:%.*]] = icmp sgt i32 [[TMP21]], 35
; CHECK-VF2IC1-NEXT:    br i1 [[CMP1]], label %if.then, label %for.inc
; CHECK-VF2IC1:       if.then:
; CHECK-VF2IC1:         [[TMP22:%.*]] = load i32, i32* {{%.*}}, align 4
; CHECK-VF2IC1-NEXT:    [[CMP3:%.*]] = icmp eq i32 [[TMP22]], 2
; CHECK-VF2IC1-NEXT:    [[SPEC_SELECT:%.*]] = select i1 [[CMP3]], i32 1, i32 [[R_012]]
; CHECK-VF2IC1-NEXT:    br label %for.inc
; CHECK-VF2IC1:       for.inc:
; CHECK-VF2IC1-NEXT:    [[R_1]] = phi i32 [ [[R_012]], %for.body ], [ [[SPEC_SELECT]], %if.then ]
; CHECK-VF2IC1:       for.end.loopexit:
; CHECK-VF2IC1-NEXT:    [[R_1_LCSSA:%.*]] = phi i32 [ [[R_1]], %for.inc ], [ [[RDX_SELECT]], %middle.block ]
; CHECK-VF2IC1-NEXT:    ret i32 [[R_1_LCSSA]]
;
; CHECK-VF1IC2-LABEL: @pred_select_const_i32_from_icmp(
; CHECK-VF1IC2:       vector.body:
; CHECK-VF1IC2:         [[VEC_PHI:%.*]] = phi i32 [ 0, %vector.ph ], [ [[PREDPHI:%.*]], %pred.load.continue4 ]
; CHECK-VF1IC2-NEXT:    [[VEC_PHI2:%.*]] = phi i32 [ 0, %vector.ph ], [ [[PREDPHI5:%.*]], %pred.load.continue4 ]
; CHECK-VF1IC2:         [[TMP0:%.*]] = getelementptr inbounds i32, i32* [[SRC1:%.*]], i64 {{%.*}}
; CHECK-VF1IC2-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, i32* [[SRC1]], i64 {{%.*}}
; CHECK-VF1IC2-NEXT:    [[TMP2:%.*]] = load i32, i32* [[TMP0]], align 4
; CHECK-VF1IC2-NEXT:    [[TMP3:%.*]] = load i32, i32* [[TMP1]], align 4
; CHECK-VF1IC2-NEXT:    [[TMP4:%.*]] = icmp sgt i32 [[TMP2]], 35
; CHECK-VF1IC2-NEXT:    [[TMP5:%.*]] = icmp sgt i32 [[TMP3]], 35
; CHECK-VF1IC2-NEXT:    br i1 [[TMP4]], label %pred.load.if, label %pred.load.continue
; CHECK-VF1IC2:       pred.load.if:
; CHECK-VF1IC2-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, i32* [[SRC2:%.*]], i64 {{%.*}}
; CHECK-VF1IC2-NEXT:    [[TMP7:%.*]] = load i32, i32* [[TMP6]], align 4
; CHECK-VF1IC2-NEXT:    br label %pred.load.continue
; CHECK-VF1IC2:       pred.load.continue:
; CHECK-VF1IC2-NEXT:    [[TMP8:%.*]] = phi i32 [ poison, %vector.body ], [ [[TMP7]], %pred.load.if ]
; CHECK-VF1IC2-NEXT:    br i1 [[TMP5]], label %pred.load.if3, label %pred.load.continue4
; CHECK-VF1IC2:       pred.load.if3:
; CHECK-VF1IC2-NEXT:    [[TMP9:%.*]] = getelementptr inbounds i32, i32* [[SRC2]], i64 {{%.*}}
; CHECK-VF1IC2-NEXT:    [[TMP10:%.*]] = load i32, i32* [[TMP9]], align 4
; CHECK-VF1IC2-NEXT:    br label %pred.load.continue4
; CHECK-VF1IC2:       pred.load.continue4:
; CHECK-VF1IC2-NEXT:    [[TMP11:%.*]] = phi i32 [ poison, %pred.load.continue ], [ [[TMP10]], %pred.load.if3 ]
; CHECK-VF1IC2-NEXT:    [[TMP12:%.*]] = icmp eq i32 [[TMP8]], 2
; CHECK-VF1IC2-NEXT:    [[TMP13:%.*]] = icmp eq i32 [[TMP11]], 2
; CHECK-VF1IC2-NEXT:    [[TMP14:%.*]] = select i1 [[TMP12]], i32 1, i32 [[VEC_PHI]]
; CHECK-VF1IC2-NEXT:    [[TMP15:%.*]] = select i1 [[TMP13]], i32 1, i32 [[VEC_PHI2]]
; CHECK-VF1IC2-NEXT:    [[TMP16:%.*]] = xor i1 [[TMP4]], true
; CHECK-VF1IC2-NEXT:    [[TMP17:%.*]] = xor i1 [[TMP5]], true
; CHECK-VF1IC2-NEXT:    [[PREDPHI]] = select i1 [[TMP4]], i32 [[TMP14]], i32 [[VEC_PHI]]
; CHECK-VF1IC2-NEXT:    [[PREDPHI5]] = select i1 [[TMP5]], i32 [[TMP15]], i32 [[VEC_PHI2]]
; CHECK-VF1IC2:         br i1 {{%.*}}, label %middle.block, label %vector.body
; CHECK-VF1IC2:       middle.block:
; CHECK-VF1IC2-NEXT:    [[RDX_SELECT_CMP:%.*]] = icmp ne i32 [[PREDPHI]], 0
; CHECK-VF1IC2-NEXT:    [[RDX_SELECT:%.*]] = select i1 [[RDX_SELECT_CMP]], i32 [[PREDPHI]], i32 [[PREDPHI5]]
; CHECK-VF1IC2:         br i1 {{%.*}}, label %for.end.loopexit, label %scalar.ph
; CHECK-VF1IC2:       scalar.ph:
; CHECK-VF1IC2-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ {{%.*}}, %middle.block ], [ 0, %entry ]
; CHECK-VF1IC2-NEXT:    [[BC_MERGE_RDX:%.*]] = phi i32 [ 0, %entry ], [ [[RDX_SELECT]], %middle.block ]
; CHECK-VF1IC2-NEXT:    br label %for.body
; CHECK-VF1IC2:       for.body:
; CHECK-VF1IC2-NEXT:    [[I_013:%.*]] = phi i64 [ [[INC:%.*]], %for.inc ], [ [[BC_RESUME_VAL]], %scalar.ph ]
; CHECK-VF1IC2-NEXT:    [[R_012:%.*]] = phi i32 [ [[R_1:%.*]], %for.inc ], [ [[BC_MERGE_RDX]], %scalar.ph ]
; CHECK-VF1IC2:         [[TMP19:%.*]] = load i32, i32* {{%.*}}, align 4
; CHECK-VF1IC2-NEXT:    [[CMP1:%.*]] = icmp sgt i32 [[TMP19]], 35
; CHECK-VF1IC2-NEXT:    br i1 [[CMP1]], label [[IF_THEN:%.*]], label %for.inc
; CHECK-VF1IC2:       if.then:
; CHECK-VF1IC2:         [[TMP20:%.*]] = load i32, i32* {{%.*}}, align 4
; CHECK-VF1IC2-NEXT:    [[CMP3:%.*]] = icmp eq i32 [[TMP20]], 2
; CHECK-VF1IC2-NEXT:    [[SPEC_SELECT:%.*]] = select i1 [[CMP3]], i32 1, i32 [[R_012]]
; CHECK-VF1IC2-NEXT:    br label %for.inc
; CHECK-VF1IC2:       for.inc:
; CHECK-VF1IC2-NEXT:    [[R_1]] = phi i32 [ [[R_012]], %for.body ], [ [[SPEC_SELECT]], %if.then ]
; CHECK-VF1IC2:         br i1 {{%.*}}, label %for.end.loopexit, label %for.body
; CHECK-VF1IC2:       for.end.loopexit:
; CHECK-VF1IC2-NEXT:    [[R_1_LCSSA:%.*]] = phi i32 [ [[R_1]], %for.inc ], [ [[RDX_SELECT]], %middle.block ]
; CHECK-VF1IC2-NEXT:    ret i32 [[R_1_LCSSA]]
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.013 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %r.012 = phi i32 [ %r.1, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %src1, i64 %i.013
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 35
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx2 = getelementptr inbounds i32, i32* %src2, i64 %i.013
  %1 = load i32, i32* %arrayidx2, align 4
  %cmp3 = icmp eq i32 %1, 2
  %spec.select = select i1 %cmp3, i32 1, i32 %r.012
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %r.1 = phi i32 [ %r.012, %for.body ], [ %spec.select, %if.then ]
  %inc = add nuw nsw i64 %i.013, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.inc
  %r.1.lcssa = phi i32 [ %r.1, %for.inc ]
  ret i32 %r.1.lcssa
}
