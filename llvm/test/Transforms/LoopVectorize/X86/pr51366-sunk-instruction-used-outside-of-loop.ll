; RUN: opt -opaque-pointers=0 -passes=loop-vectorize -mtriple=x86_64-unknown-linux -force-vector-width=2 -force-vector-interleave=1 -S %s | FileCheck %s

define i32* @test(i32* noalias %src, i32* noalias %dst) {
; CHECK-LABEL: @test(
; CHECK-NEXT:  entry:
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], [[PRED_LOAD_CONTINUE2:%.*]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph ], [ [[VEC_IND_NEXT:%.*]], [[PRED_LOAD_CONTINUE2]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[INDEX]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32, i32* [[SRC:%.*]], i64 [[TMP1]]
; CHECK-NEXT:    [[TMP3:%.*]] = icmp eq <2 x i64> [[VEC_IND]], zeroinitializer
; CHECK-NEXT:    [[TMP4:%.*]] = xor <2 x i1> [[TMP3]], <i1 true, i1 true>
; CHECK-NEXT:    [[TMP5:%.*]] = extractelement <2 x i1> [[TMP4]], i32 0
; CHECK-NEXT:    br i1 [[TMP5]], label [[PRED_LOAD_IF:%.*]], label [[PRED_LOAD_CONTINUE:%.*]]
; CHECK:       pred.load.if:
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, i32* [[SRC]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP7:%.*]] = load i32, i32* [[TMP6]], align 4
; CHECK-NEXT:    [[TMP8:%.*]] = insertelement <2 x i32> poison, i32 [[TMP7]], i32 0
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE]]
; CHECK:       pred.load.continue:
; CHECK-NEXT:    [[TMP9:%.*]] = phi <2 x i32> [ poison, %vector.body ], [ [[TMP8]], [[PRED_LOAD_IF]] ]
; CHECK-NEXT:    [[TMP10:%.*]] = extractelement <2 x i1> [[TMP4]], i32 1
; CHECK-NEXT:    br i1 [[TMP10]], label [[PRED_LOAD_IF1:%.*]], label [[PRED_LOAD_CONTINUE2]]
; CHECK:       pred.load.if1:
; CHECK-NEXT:    [[TMP11:%.*]] = load i32, i32* [[TMP2]], align 4
; CHECK-NEXT:    [[TMP12:%.*]] = insertelement <2 x i32> [[TMP9]], i32 [[TMP11]], i32 1
; CHECK-NEXT:    br label [[PRED_LOAD_CONTINUE2]]
; CHECK:       pred.load.continue2:
; CHECK-NEXT:    [[TMP13:%.*]] = phi <2 x i32> [ [[TMP9]], [[PRED_LOAD_CONTINUE]] ], [ [[TMP12]], [[PRED_LOAD_IF1]] ]
; CHECK-NEXT:    [[PREDPHI:%.*]] = select <2 x i1> [[TMP3]], <2 x i32> zeroinitializer, <2 x i32> [[TMP13]]
; CHECK-NEXT:    [[TMP14:%.*]] = getelementptr inbounds i32, i32* [[DST:%.*]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP15:%.*]] = getelementptr inbounds i32, i32* [[TMP14]], i32 0
; CHECK-NEXT:    [[TMP16:%.*]] = bitcast i32* [[TMP15]] to <2 x i32>*
; CHECK-NEXT:    store <2 x i32> [[PREDPHI]], <2 x i32>* [[TMP16]], align 4
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <2 x i64> [[VEC_IND]], <i64 2, i64 2>
; CHECK-NEXT:    [[TMP17:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP17]], label [[MIDDLE_BLOCK:%.*]], label %vector.body
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i64 1000, 1000
; CHECK-NEXT:    br i1 [[CMP_N]], label %exit, label %scalar.ph
; CHECK:       exit:
; CHECK-NEXT:    [[GEP_LCSSA:%.*]] = phi i32* [ %gep.src, %loop.latch ], [ [[TMP2]], %middle.block ]
; CHECK-NEXT:    ret i32* [[GEP_LCSSA]]
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.src = getelementptr inbounds i32, i32* %src, i64 %iv
  %cmp.1 = icmp eq i64 %iv, 0
  br i1 %cmp.1, label %loop.latch, label %then

then:
  %l = load i32, i32* %gep.src, align 4
  br label %loop.latch

loop.latch:
  %m = phi i32 [ %l, %then ], [ 0, %loop.header ]
  %gep.dst = getelementptr inbounds i32, i32* %dst, i64 %iv
  store i32 %m, i32* %gep.dst, align 4
  %iv.next = add nsw i64 %iv, 1
  %cmp.2 = icmp slt i64 %iv.next, 1000
  br i1 %cmp.2, label %loop.header, label %exit

exit:
  %gep.lcssa = phi i32* [ %gep.src, %loop.latch ]
  ret i32* %gep.lcssa
}
