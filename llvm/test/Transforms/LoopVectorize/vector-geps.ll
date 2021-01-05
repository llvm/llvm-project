; RUN: opt < %s -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -instcombine -S | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: @vector_gep_stored(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, i32* %b, <4 x i64> [[VEC_IND]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32*, i32** %a, i64 [[INDEX]]
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast i32** [[TMP2]] to <4 x i32*>*
; CHECK-NEXT:    store <4 x i32*> [[TMP1]], <4 x i32*>* [[TMP3]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 4
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <4 x i64> [[VEC_IND]], <i64 4, i64 4, i64 4, i64 4>
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @vector_gep_stored(i32** %a, i32 *%b, i64 %n) {
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

; CHECK-LABEL: @uniform_vector_gep_stored(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, i32* %b, i64 1
; CHECK-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <4 x i32*> poison, i32* [[TMP1]], i32 0
; CHECK-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <4 x i32*> [[DOTSPLATINSERT]], <4 x i32*> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32*, i32** %a, i64 [[INDEX]]
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast i32** [[TMP2]] to <4 x i32*>*
; CHECK-NEXT:    store <4 x i32*> [[DOTSPLAT]], <4 x i32*>* [[TMP3]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 4
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @uniform_vector_gep_stored(i32** %a, i32 *%b, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %tmp0 = getelementptr inbounds i32, i32* %b, i64 1
  %tmp1 = getelementptr inbounds i32*, i32** %a, i64 %i
  store i32* %tmp0, i32** %tmp1, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}
