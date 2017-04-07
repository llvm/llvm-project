; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -instcombine -S -debug-only=loop-vectorize -disable-output -print-after=instcombine 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: PR31671
;
; Check a pointer in which one of its uses is consecutive-like and another of
; its uses is non-consecutive-like. In the test case below, %tmp3 is the
; pointer operand of an interleaved load, making it consecutive-like. However,
; it is also the pointer operand of a non-interleaved store that will become a
; scatter operation. %tmp3 (and the induction variable) should not be marked
; uniform-after-vectorization.
;
; CHECK:       LV: Found uniform instruction: %tmp0 = getelementptr inbounds %data, %data* %d, i64 0, i32 3, i64 %i
; CHECK-NOT:   LV: Found uniform instruction: %tmp3 = getelementptr inbounds %data, %data* %d, i64 0, i32 0, i64 %i
; CHECK-NOT:   LV: Found uniform instruction: %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
; CHECK-NOT:   LV: Found uniform instruction: %i.next = add nuw nsw i64 %i, 5
; CHECK:       vector.ph:
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <16 x float> undef, float %x, i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <16 x float> [[BROADCAST_SPLATINSERT]], <16 x float> undef, <16 x i32> zeroinitializer
; CHECK-NEXT:    br label %vector.body
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <16 x i64> [ <i64 0, i64 5, i64 10, i64 15, i64 20, i64 25, i64 30, i64 35, i64 40, i64 45, i64 50, i64 55, i64 60, i64 65, i64 70, i64 75>, %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[OFFSET_IDX:%.*]] = mul i64 [[INDEX]], 5
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds %data, %data* %d, i64 0, i32 3, i64 [[OFFSET_IDX]]
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast float* [[TMP0]] to <80 x float>*
; CHECK-NEXT:    [[WIDE_VEC:%.*]] = load <80 x float>, <80 x float>* [[TMP1]], align 4
; CHECK-NEXT:    [[STRIDED_VEC:%.*]] = shufflevector <80 x float> [[WIDE_VEC]], <80 x float> undef, <16 x i32> <i32 0, i32 5, i32 10, i32 15, i32 20, i32 25, i32 30, i32 35, i32 40, i32 45, i32 50, i32 55, i32 60, i32 65, i32 70, i32 75>
; CHECK-NEXT:    [[TMP2:%.*]] = fmul <16 x float> [[BROADCAST_SPLAT]], [[STRIDED_VEC]]
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds %data, %data* %d, i64 0, i32 0, <16 x i64> [[VEC_IND]]
; CHECK-NEXT:    [[BC:%.*]] = bitcast <16 x float*> [[TMP3]] to <16 x <80 x float>*>
; CHECK-NEXT:    [[TMP4:%.*]] = extractelement <16 x <80 x float>*> [[BC]], i32 0
; CHECK-NEXT:    [[WIDE_VEC1:%.*]] = load <80 x float>, <80 x float>* [[TMP4]], align 4
; CHECK-NEXT:    [[STRIDED_VEC2:%.*]] = shufflevector <80 x float> [[WIDE_VEC1]], <80 x float> undef, <16 x i32> <i32 0, i32 5, i32 10, i32 15, i32 20, i32 25, i32 30, i32 35, i32 40, i32 45, i32 50, i32 55, i32 60, i32 65, i32 70, i32 75>
; CHECK-NEXT:    [[TMP5:%.*]] = fadd <16 x float> [[STRIDED_VEC2]], [[TMP2]]
; CHECK-NEXT:    call void @llvm.masked.scatter.v16f32(<16 x float> [[TMP5]], <16 x float*> [[TMP3]], i32 4, <16 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 16
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <16 x i64> [[VEC_IND]], <i64 80, i64 80, i64 80, i64 80, i64 80, i64 80, i64 80, i64 80, i64 80, i64 80, i64 80, i64 80, i64 80, i64 80, i64 80, i64 80>
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body

%data = type { [32000 x float], [3 x i32], [4 x i8], [32000 x float] }

define void @PR31671(float %x, %data* %d) #0 {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %tmp0 = getelementptr inbounds %data, %data* %d, i64 0, i32 3, i64 %i
  %tmp1 = load float, float* %tmp0, align 4
  %tmp2 = fmul float %x, %tmp1
  %tmp3 = getelementptr inbounds %data, %data* %d, i64 0, i32 0, i64 %i
  %tmp4 = load float, float* %tmp3, align 4
  %tmp5 = fadd float %tmp4, %tmp2
  store float %tmp5, float* %tmp3, align 4
  %i.next = add nuw nsw i64 %i, 5
  %cond = icmp slt i64 %i.next, 32000
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

attributes #0 = { "target-cpu"="knl" }
