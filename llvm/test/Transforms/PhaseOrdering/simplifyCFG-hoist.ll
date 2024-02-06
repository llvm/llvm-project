; opt -passes='default<O3>' -S --mtriple=aarch64-linux-gnu --mcpu=a64fx  < %s  | FileCheck %s

; Hoist identical instructions from successor blocks even if
; they are not located at the same level. This could help generate
; more compact vectorized code.
; More info can be found at https://github.com/llvm/llvm-project/issues/68395.


define void @hoist_then_vectorize(ptr %a, ptr %b, ptr %c, ptr %d, i32 %N){
; CHECK-LABEL: @hoist_then_vectorize(
; CHECK-NEXT:  iter.check:
; CHECK-NEXT:    [[VSCALE:%.*]] = tail call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[SHIFT:%.*]] = shl i64 [[VSCALE:%.*]], 1
; CHECK-NEXT:    [[MIN_ITR:%.*]] = icmp ugt i64 [[SHIFT:%.*]], 20
; CHECK-NEXT:    br i1 [[MIN_ITR:%.*]], label [[FOR_BODY_PREHEADER:%.*]], label [[VECTOR_MAIN_LOOP_ITR_CHECK:%.*]] 
; CHECK:       vector.main.loop.iter.check:
; CHECK-NEXT:    [[VSCALE2:%.*]] = tail call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[SHIFT2:%.*]] = shl i64 [[VSCALE2:%.*]], 2
; CHECK-NEXT:    [[MIN_ITR2:%.*]] = icmp ugt i64 [[SHIFT2:%.*]], 20
; CHECK-NEXT:    br i1 [[MIN_ITR2:%.*]], label [[VEC_EPILOG_PH:%.*]], label [[VECTOR_PH:%.*]] 
; CHECK:       vector.ph: 
; CHECK-NEXT:    [[VSCALE3:%.*]] = tail call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[SHIFT3:%.*]] = shl i64 [[VSCALE3:%.*]], 2
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 20, [[SHIFT3:%.*]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub nuw nsw i64 20, [[N_MOD_VF:%.*]]
; CHECK-NEXT:    [[VSCALE4:%.*]] = tail call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[SHIFT4:%.*]] = shl i64 [[VSCALE4:%.*]], 2
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]] 
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH:%.*]]  ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY:%.*]] ]
; CHECK-NEXT:    [[GEP_D:%.*]]  = getelementptr inbounds i32, ptr [[D:%.*]], i64 [[INDEX:%.*]]
; CHECK-NEXT:    [[LOAD_D:%.*]] = load <vscale x 4 x i32>, ptr [[GEP_D:%.*]], align 4
; CHECK-NEXT:    [[MASK1:%.*]] = icmp slt <vscale x 4 x i32> [[LOAD_D:%.*]], zeroinitializer
; CHECK-NEXT:    [[GEP_A:%.*]]  = getelementptr inbounds i32, ptr [[A:%.*]], i64 [[INDEX:%.*]]
; CHECK-NEXT:    [[LOAD_A:%.*]] = load <vscale x 4 x i32>, ptr [[GEP_A:%.*]], align 4
; CHECK-NEXT:    [[MASK2:%.*]] = icmp eq <vscale x 4 x i32> [[LOAD_A:%.*]], zeroinitializer
; CHECK-NEXT:    [[SEL1:%.*]] = select <vscale x 4 x i1> [[MASK2:%.*]], <vscale x 4 x i32> shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 2, i64 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i32> shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 3, i64 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[SEL2:%.*]] = select <vscale x 4 x i1> [[MASK1:%.*]], <vscale x 4 x i32> shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 1, i64 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i32> [[SEL1:%.*]]
; CHECK-NEXT:    [[ADD:%.*]] = add <vscale x 4 x i32> [[LOAD_A:%.*]], [[SEL2:%.*]]
; CHECK-NEXT:    store <vscale x 4 x i32> [[ADD:%.*]], ptr [[GEP_A:%.*]], align 4
; CHECK-NEXT:    [[INDEX_NEXT:%.*]] = add nuw i64 [[INDEX:%.*]], [[SHIFT4:%.*]]
; CHECK-NEXT:    [[LOOP_COND:%.*]] = icmp eq i64 [[INDEX_NEXT:%.*]], [[N_VEC:%.*]]
; CHECK-NEXT:    br i1 [[LOOP_COND:%.*]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY:%.*]] 

entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc
  ret void

for.body:                                         ; preds = %entry, %for.inc
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds i32, ptr %d, i64 %indvars.iv
  %ldr_d = load i32, ptr %arrayidx, align 4 
  %cmp1 = icmp slt i32 %ldr_d, 0
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %arrayidx3 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %ldr_a = load i32, ptr %arrayidx3, align 4
  %add33 = add i32 %ldr_a, 1
  store i32 %add33, ptr %arrayidx3, align 4
  br label %for.inc

if.else:                                          ; preds = %for.body
  %cmp7 = icmp eq i32 %ldr_d, 0
  br i1 %cmp7, label %if.then9, label %if.else15

if.then9:                                         ; preds = %if.else
  %arrayidx11 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %ldr_a2 = load i32, ptr %arrayidx11, align 4
  %add1334 = add i32 %ldr_a2, 2
  store i32 %add1334, ptr %arrayidx11, align 4
  br label %for.inc

if.else15:                                        ; preds = %if.else
  %arrayidx112 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %ldr_a3 = load i32, ptr %arrayidx112, align 4
  %add1935 = add i32 %ldr_a3, 3
  store i32 %add1935, ptr %arrayidx112, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.then, %if.else15, %if.then9
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 20
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}