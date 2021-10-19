; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -force-vector-width=2 -force-vector-interleave=1 -instcombine -debug-only=loop-vectorize -disable-output -print-after=instcombine -enable-new-pm=0 2>&1 | FileCheck %s
; RUN: opt < %s -aa-pipeline=basic-aa -passes=loop-vectorize,instcombine -force-vector-width=2 -force-vector-interleave=1 -debug-only=loop-vectorize -disable-output -print-after=instcombine 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: vector_gep
; CHECK-NOT:   LV: Found scalar instruction: %tmp0 = getelementptr inbounds i32, i32* %b, i64 %i
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, i32* %b, <2 x i64> [[VEC_IND]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32*, i32** %a, i64 [[INDEX]]
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast i32** [[TMP2]] to <2 x i32*>*
; CHECK-NEXT:    store <2 x i32*> [[TMP1]], <2 x i32*>* [[TMP3]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <2 x i64> [[VEC_IND]], <i64 2, i64 2>
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @vector_gep(i32** %a, i32 *%b, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %tmp0 = getelementptr inbounds i32, i32* %b, i64 %i
  %tmp1 = getelementptr inbounds i32*, i32** %a, i64 %i
  store i32* %tmp0, i32** %tmp1, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: scalar_store
; CHECK:       LV: Found scalar instruction: %tmp1 = getelementptr inbounds i32*, i32** %a, i64 %i
; CHECK-NEXT:  LV: Found scalar instruction: %tmp0 = getelementptr inbounds i32, i32* %b, i64 %i
; CHECK-NEXT:  LV: Found scalar instruction: %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:  LV: Found scalar instruction: %i.next = add nuw nsw i64 %i, 2
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[OFFSET_IDX:%.*]] = shl i64 [[INDEX]], 1
; CHECK-NEXT:    [[TMP4:%.*]] = or i64 [[OFFSET_IDX]], 2
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i32, i32* %b, i64 [[OFFSET_IDX]]
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, i32* %b, i64 [[TMP4]]
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32*, i32** %a, i64 [[OFFSET_IDX]]
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32*, i32** %a, i64 [[TMP4]]
; CHECK-NEXT:    store i32* [[TMP5]], i32** [[TMP7]], align 8
; CHECK-NEXT:    store i32* [[TMP6]], i32** [[TMP8]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @scalar_store(i32** %a, i32 *%b, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %tmp0 = getelementptr inbounds i32, i32* %b, i64 %i
  %tmp1 = getelementptr inbounds i32*, i32** %a, i64 %i
  store i32* %tmp0, i32** %tmp1, align 8
  %i.next = add nuw nsw i64 %i, 2
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: expansion
; CHECK:       LV: Found scalar instruction: %tmp3 = getelementptr inbounds i32*, i32** %tmp2, i64 %i
; CHECK-NEXT:  LV: Found scalar instruction: %tmp1 = bitcast i64* %tmp0 to i32*
; CHECK-NEXT:  LV: Found scalar instruction: %tmp2 = getelementptr inbounds i32*, i32** %a, i64 0
; CHECK-NEXT:  LV: Found scalar instruction: %tmp0 = getelementptr inbounds i64, i64* %b, i64 %i
; CHECK-NEXT:  LV: Found scalar instruction: %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:  LV: Found scalar instruction: %i.next = add nuw nsw i64 %i, 2
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[OFFSET_IDX:%.*]] = shl i64 [[INDEX]], 1
; CHECK-NEXT:    [[TMP4:%.*]] = or i64 [[OFFSET_IDX]], 2
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i64, i64* %b, i64 [[OFFSET_IDX]]
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i64, i64* %b, i64 [[TMP4]]
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32*, i32** %a, i64 [[OFFSET_IDX]]
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32*, i32** %a, i64 [[TMP4]]
; CHECK-NEXT:    [[TMP9:%.*]] = bitcast i32** [[TMP7]] to i64**
; CHECK-NEXT:    store i64* [[TMP5]], i64** [[TMP9]], align 8
; CHECK-NEXT:    [[TMP10:%.*]] = bitcast i32** [[TMP8]] to i64**
; CHECK-NEXT:    store i64* [[TMP6]], i64** [[TMP10]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @expansion(i32** %a, i64 *%b, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %tmp0 = getelementptr inbounds i64, i64* %b, i64 %i
  %tmp1 = bitcast i64* %tmp0 to i32*
  %tmp2 = getelementptr inbounds i32*, i32** %a, i64 0
  %tmp3 = getelementptr inbounds i32*, i32** %tmp2, i64 %i
  store i32* %tmp1, i32** %tmp3, align 8
  %i.next = add nuw nsw i64 %i, 2
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; CHECK-LABEL: no_gep_or_bitcast
; CHECK-NOT:   LV: Found scalar instruction: %tmp1 = load i32*, i32** %tmp0, align 8
; CHECK:       LV: Found scalar instruction: %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
; CHECK-NEXT:  LV: Found scalar instruction: %i.next = add nuw nsw i64 %i, 1
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32*, i32** %a, i64 [[INDEX]]
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast i32** [[TMP1]] to <2 x i32*>*
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <2 x i32*>, <2 x i32*>* [[TMP2]], align 8
; CHECK-NEXT:    [[TMP3:%.*]] = extractelement <2 x i32*> [[WIDE_LOAD]], i32 0
; CHECK-NEXT:    store i32 0, i32* [[TMP3]], align 8
; CHECK-NEXT:    [[TMP4:%.*]] = extractelement <2 x i32*> [[WIDE_LOAD]], i32 1
; CHECK-NEXT:    store i32 0, i32* [[TMP4]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @no_gep_or_bitcast(i32** noalias %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %tmp0 = getelementptr inbounds i32*, i32** %a, i64 %i
  %tmp1 = load i32*, i32** %tmp0, align 8
  store i32 0, i32* %tmp1, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}
