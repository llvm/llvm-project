; RUN: opt -basicaa -loop-accesses -analyze < %s | FileCheck %s -check-prefix=LAA
; RUN: opt -passes='require<aa>,require<scalar-evolution>,require<aa>,loop(print-access-info)' -aa-pipeline='basic-aa' -disable-output < %s  2>&1 | FileCheck %s --check-prefix=LAA
; RUN: opt -loop-versioning -S < %s | FileCheck %s -check-prefix=LV

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; For this loop:
;   unsigned index = 0;
;   for (int i = 0; i < n; i++) {
;    A[2 * index] = A[2 * index] + B[i];
;    index++;
;   }
;
; SCEV is unable to prove that A[2 * i] does not overflow.
;
; Analyzing the IR does not help us because the GEPs are not
; affine AddRecExprs. However, we can turn them into AddRecExprs
; using SCEV Predicates.
;
; Once we have an affine expression we need to add an additional NUSW
; to check that the pointers don't wrap since the GEPs are not
; inbound.

; LAA-LABEL: f1
; LAA: Memory dependences are safe{{$}}
; LAA: SCEV assumptions:
; LAA-NEXT: {0,+,2}<%for.body> Added Flags: <nusw>
; LAA-NEXT: {%a,+,4}<%for.body> Added Flags: <nusw>

; The expression for %mul_ext as analyzed by SCEV is
;    (zext i32 {0,+,2}<%for.body> to i64)
; We have added the nusw flag to turn this expression into the SCEV expression:
;    i64 {0,+,2}<%for.body>

; LAA: [PSE]  %arrayidxA = getelementptr i16, i16* %a, i64 %mul_ext:
; LAA-NEXT: ((2 * (zext i32 {0,+,2}<%for.body> to i64)) + %a)
; LAA-NEXT: --> {%a,+,4}<%for.body>


; LV-LABEL: f1
; LV-LABEL: for.body.lver.check

; LV:      [[BETrunc:%[^ ]*]] = trunc i64 [[BE:%[^ ]*]] to i32
; LV-NEXT: [[OFMul:%[^ ]*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 2, i32 [[BETrunc]])
; LV-NEXT: [[OFMulResult:%[^ ]*]] = extractvalue { i32, i1 } [[OFMul]], 0
; LV-NEXT: [[OFMulOverflow:%[^ ]*]] = extractvalue { i32, i1 } [[OFMul]], 1
; LV-NEXT: [[AddEnd:%[^ ]*]] = add i32 0, [[OFMulResult]]
; LV-NEXT: [[SubEnd:%[^ ]*]] = sub i32 0, [[OFMulResult]]
; LV-NEXT: [[CmpNeg:%[^ ]*]] = icmp ugt i32 [[SubEnd]], 0
; LV-NEXT: [[CmpPos:%[^ ]*]] = icmp ult i32 [[AddEnd]], 0
; LV-NEXT: [[Cmp:%[^ ]*]] = select i1 false, i1 [[CmpNeg]], i1 [[CmpPos]]
; LV-NEXT: [[BECheck:%[^ ]*]] = icmp ugt i64 [[BE]], 4294967295
; LV-NEXT: [[CheckOr0:%[^ ]*]] = or i1 [[Cmp]], [[BECheck]]
; LV-NEXT: [[PredCheck0:%[^ ]*]] = or i1 [[CheckOr0]], [[OFMulOverflow]]

; LV-NEXT: [[Or0:%[^ ]*]] = or i1 false, [[PredCheck0]]

; LV-NEXT: [[OFMul1:%[^ ]*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 4, i64 [[BE]])
; LV-NEXT: [[OFMulResult1:%[^ ]*]] = extractvalue { i64, i1 } [[OFMul1]], 0
; LV-NEXT: [[OFMulOverflow1:%[^ ]*]] = extractvalue { i64, i1 } [[OFMul1]], 1
; LV-NEXT: [[AddEnd1:%[^ ]*]] = add i64 %a2, [[OFMulResult1]]
; LV-NEXT: [[SubEnd1:%[^ ]*]] = sub i64 %a2, [[OFMulResult1]]
; LV-NEXT: [[CmpNeg1:%[^ ]*]] = icmp ugt i64 [[SubEnd1]], %a2
; LV-NEXT: [[CmpPos1:%[^ ]*]] = icmp ult i64 [[AddEnd1]], %a2
; LV-NEXT: [[Cmp:%[^ ]*]] = select i1 false, i1 [[CmpNeg1]], i1 [[CmpPos1]]
; LV-NEXT: [[PredCheck1:%[^ ]*]] = or i1 [[Cmp]], [[OFMulOverflow1]]

; LV: [[FinalCheck:%[^ ]*]] = or i1 [[Or0]], [[PredCheck1]]
; LV: br i1 [[FinalCheck]], label %for.body.ph.lver.orig, label %for.body.ph
define void @f1(i16* noalias %a,
                i16* noalias %b, i64 %N) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %ind1 = phi i32 [ 0, %entry ], [ %inc1, %for.body ]

  %mul = mul i32 %ind1, 2
  %mul_ext = zext i32 %mul to i64

  %arrayidxA = getelementptr i16, i16* %a, i64 %mul_ext
  %loadA = load i16, i16* %arrayidxA, align 2

  %arrayidxB = getelementptr i16, i16* %b, i64 %ind
  %loadB = load i16, i16* %arrayidxB, align 2

  %add = mul i16 %loadA, %loadB

  store i16 %add, i16* %arrayidxA, align 2

  %inc = add nuw nsw i64 %ind, 1
  %inc1 = add i32 %ind1, 1

  %exitcond = icmp eq i64 %inc, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; For this loop:
;   unsigned index = n;
;   for (int i = 0; i < n; i++) {
;    A[2 * index] = A[2 * index] + B[i];
;    index--;
;   }
;
; the SCEV expression for 2 * index is not an AddRecExpr
; (and implictly not affine). However, we are able to make assumptions
; that will turn the expression into an affine one and continue the
; analysis.
;
; Once we have an affine expression we need to add an additional NUSW
; to check that the pointers don't wrap since the GEPs are not
; inbounds.
;
; This loop has a negative stride for A, and the nusw flag is required in
; order to properly extend the increment from i32 -4 to i64 -4.

; LAA-LABEL: f2
; LAA: Memory dependences are safe{{$}}
; LAA: SCEV assumptions:
; LAA-NEXT: {(2 * (trunc i64 %N to i32)),+,-2}<%for.body> Added Flags: <nusw>
; LAA-NEXT: {((4 * (zext i31 (trunc i64 %N to i31) to i64)) + %a),+,-4}<%for.body> Added Flags: <nusw>

; The expression for %mul_ext as analyzed by SCEV is
;     (zext i32 {(2 * (trunc i64 %N to i32)),+,-2}<%for.body> to i64)
; We have added the nusw flag to turn this expression into the following SCEV:
;     i64 {zext i32 (2 * (trunc i64 %N to i32)) to i64,+,-2}<%for.body>

; LAA: [PSE]  %arrayidxA = getelementptr i16, i16* %a, i64 %mul_ext:
; LAA-NEXT: ((2 * (zext i32 {(2 * (trunc i64 %N to i32)),+,-2}<%for.body> to i64)) + %a)
; LAA-NEXT: --> {((4 * (zext i31 (trunc i64 %N to i31) to i64)) + %a),+,-4}<%for.body>

; LV-LABEL: f2
; LV-LABEL: for.body.lver.check

; LV: [[OFMul:%[^ ]*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 2, i32 [[BETrunc:%[^ ]*]])
; LV-NEXT: [[OFMulResult:%[^ ]*]] = extractvalue { i32, i1 } [[OFMul]], 0
; LV-NEXT: [[OFMulOverflow:%[^ ]*]] = extractvalue { i32, i1 } [[OFMul]], 1
; LV-NEXT: [[AddEnd:%[^ ]*]] = add i32 [[Start:%[^ ]*]], [[OFMulResult]]
; LV-NEXT: [[SubEnd:%[^ ]*]] = sub i32 [[Start]], [[OFMulResult]]
; LV-NEXT: [[CmpNeg:%[^ ]*]] = icmp ugt i32 [[SubEnd]], [[Start]]
; LV-NEXT: [[CmpPos:%[^ ]*]] = icmp ult i32 [[AddEnd]], [[Start]]
; LV-NEXT: [[Cmp:%[^ ]*]] = select i1 true, i1 [[CmpNeg]], i1 [[CmpPos]]
; LV-NEXT: [[BECheck:%[^ ]*]] = icmp ugt i64 [[BE]], 4294967295
; LV-NEXT: [[CheckOr0:%[^ ]*]] = or i1 [[Cmp]], [[BECheck]]
; LV-NEXT: [[PredCheck0:%[^ ]*]] = or i1 [[CheckOr0]], [[OFMulOverflow]]

; LV-NEXT: [[Or0:%[^ ]*]] = or i1 false, [[PredCheck0]]

; LV: [[OFMul1:%[^ ]*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 4, i64 [[BE]])
; LV-NEXT: [[OFMulResult1:%[^ ]*]] = extractvalue { i64, i1 } [[OFMul1]], 0
; LV-NEXT: [[OFMulOverflow1:%[^ ]*]] = extractvalue { i64, i1 } [[OFMul1]], 1
; LV-NEXT: [[AddEnd1:%[^ ]*]] = add i64 [[Start:%[^ ]*]], [[OFMulResult1]]
; LV-NEXT: [[SubEnd1:%[^ ]*]] = sub i64 [[Start]], [[OFMulResult1]]
; LV-NEXT: [[CmpNeg1:%[^ ]*]] = icmp ugt i64 [[SubEnd1]], [[Start]]
; LV-NEXT: [[CmpPos1:%[^ ]*]] = icmp ult i64 [[AddEnd1]], [[Start]]
; LV-NEXT: [[Cmp:%[^ ]*]] = select i1 true, i1 [[CmpNeg1]], i1 [[CmpPos1]]
; LV-NEXT: [[PredCheck1:%[^ ]*]] = or i1 [[Cmp]], [[OFMulOverflow1]]

; LV: [[FinalCheck:%[^ ]*]] = or i1 [[Or0]], [[PredCheck1]]
; LV: br i1 [[FinalCheck]], label %for.body.ph.lver.orig, label %for.body.ph
define void @f2(i16* noalias %a,
                i16* noalias %b, i64 %N) {
entry:
  %TruncN = trunc i64 %N to i32
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %ind1 = phi i32 [ %TruncN, %entry ], [ %dec, %for.body ]

  %mul = mul i32 %ind1, 2
  %mul_ext = zext i32 %mul to i64

  %arrayidxA = getelementptr i16, i16* %a, i64 %mul_ext
  %loadA = load i16, i16* %arrayidxA, align 2

  %arrayidxB = getelementptr i16, i16* %b, i64 %ind
  %loadB = load i16, i16* %arrayidxB, align 2

  %add = mul i16 %loadA, %loadB

  store i16 %add, i16* %arrayidxA, align 2

  %inc = add nuw nsw i64 %ind, 1
  %dec = sub i32 %ind1, 1

  %exitcond = icmp eq i64 %inc, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; We replicate the tests above, but this time sign extend 2 * index instead
; of zero extending it.

; LAA-LABEL: f3
; LAA: Memory dependences are safe{{$}}
; LAA: SCEV assumptions:
; LAA-NEXT: {0,+,2}<%for.body> Added Flags: <nssw>
; LAA-NEXT: {%a,+,4}<%for.body> Added Flags: <nusw>

; The expression for %mul_ext as analyzed by SCEV is
;     i64 (sext i32 {0,+,2}<%for.body> to i64)
; We have added the nssw flag to turn this expression into the following SCEV:
;     i64 {0,+,2}<%for.body>

; LAA: [PSE]  %arrayidxA = getelementptr i16, i16* %a, i64 %mul_ext:
; LAA-NEXT: ((2 * (sext i32 {0,+,2}<%for.body> to i64)) + %a)
; LAA-NEXT: --> {%a,+,4}<%for.body>

; LV-LABEL: f3
; LV-LABEL: for.body.lver.check

; LV: [[OFMul:%[^ ]*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 2, i32 [[BETrunc:%[^ ]*]])
; LV-NEXT: [[OFMulResult:%[^ ]*]] = extractvalue { i32, i1 } [[OFMul]], 0
; LV-NEXT: [[OFMulOverflow:%[^ ]*]] = extractvalue { i32, i1 } [[OFMul]], 1
; LV-NEXT: [[AddEnd:%[^ ]*]] = add i32 0, [[OFMulResult]]
; LV-NEXT: [[SubEnd:%[^ ]*]] = sub i32 0, [[OFMulResult]]
; LV-NEXT: [[CmpNeg:%[^ ]*]] = icmp sgt i32 [[SubEnd]], 0
; LV-NEXT: [[CmpPos:%[^ ]*]] = icmp slt i32 [[AddEnd]], 0
; LV-NEXT: [[Cmp:%[^ ]*]] = select i1 false, i1 [[CmpNeg]], i1 [[CmpPos]]
; LV-NEXT: [[BECheck:%[^ ]*]] = icmp ugt i64 [[BE]], 4294967295
; LV-NEXT: [[CheckOr0:%[^ ]*]] = or i1 [[Cmp]], [[BECheck]]
; LV-NEXT: [[PredCheck0:%[^ ]*]] = or i1 [[CheckOr0]], [[OFMulOverflow]]

; LV-NEXT: [[Or0:%[^ ]*]] = or i1 false, [[PredCheck0]]

; LV: [[OFMul1:%[^ ]*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 4, i64 [[BE:%[^ ]*]])
; LV-NEXT: [[OFMulResult1:%[^ ]*]] = extractvalue { i64, i1 } [[OFMul1]], 0
; LV-NEXT: [[OFMulOverflow1:%[^ ]*]] = extractvalue { i64, i1 } [[OFMul1]], 1
; LV-NEXT: [[AddEnd1:%[^ ]*]] = add i64 %a2, [[OFMulResult1]]
; LV-NEXT: [[SubEnd1:%[^ ]*]] = sub i64 %a2, [[OFMulResult1]]
; LV-NEXT: [[CmpNeg1:%[^ ]*]] = icmp ugt i64 [[SubEnd1]], %a2
; LV-NEXT: [[CmpPos1:%[^ ]*]] = icmp ult i64 [[AddEnd1]], %a2
; LV-NEXT: [[Cmp:%[^ ]*]] = select i1 false, i1 [[CmpNeg1]], i1 [[CmpPos1]]
; LV-NEXT: [[PredCheck1:%[^ ]*]] = or i1 [[Cmp]], [[OFMulOverflow1]]

; LV: [[FinalCheck:%[^ ]*]] = or i1 [[Or0]], [[PredCheck1]]
; LV: br i1 [[FinalCheck]], label %for.body.ph.lver.orig, label %for.body.ph
define void @f3(i16* noalias %a,
                i16* noalias %b, i64 %N) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %ind1 = phi i32 [ 0, %entry ], [ %inc1, %for.body ]

  %mul = mul i32 %ind1, 2
  %mul_ext = sext i32 %mul to i64

  %arrayidxA = getelementptr i16, i16* %a, i64 %mul_ext
  %loadA = load i16, i16* %arrayidxA, align 2

  %arrayidxB = getelementptr i16, i16* %b, i64 %ind
  %loadB = load i16, i16* %arrayidxB, align 2

  %add = mul i16 %loadA, %loadB

  store i16 %add, i16* %arrayidxA, align 2

  %inc = add nuw nsw i64 %ind, 1
  %inc1 = add i32 %ind1, 1

  %exitcond = icmp eq i64 %inc, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; LAA-LABEL: f4
; LAA: Memory dependences are safe{{$}}
; LAA: SCEV assumptions:
; LAA-NEXT: {(2 * (trunc i64 %N to i32)),+,-2}<%for.body> Added Flags: <nssw>
; LAA-NEXT: {((2 * (sext i32 (2 * (trunc i64 %N to i32)) to i64)) + %a),+,-4}<%for.body> Added Flags: <nusw>

; The expression for %mul_ext as analyzed by SCEV is
;     i64  (sext i32 {(2 * (trunc i64 %N to i32)),+,-2}<%for.body> to i64)
; We have added the nssw flag to turn this expression into the following SCEV:
;     i64 {sext i32 (2 * (trunc i64 %N to i32)) to i64,+,-2}<%for.body>

; LAA: [PSE]  %arrayidxA = getelementptr i16, i16* %a, i64 %mul_ext:
; LAA-NEXT: ((2 * (sext i32 {(2 * (trunc i64 %N to i32)),+,-2}<%for.body> to i64)) + %a)
; LAA-NEXT: --> {((2 * (sext i32 (2 * (trunc i64 %N to i32)) to i64)) + %a),+,-4}<%for.body>

; LV-LABEL: f4
; LV-LABEL: for.body.lver.check

; LV: [[OFMul:%[^ ]*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 2, i32 [[BETrunc:%[^ ]*]])
; LV-NEXT: [[OFMulResult:%[^ ]*]] = extractvalue { i32, i1 } [[OFMul]], 0
; LV-NEXT: [[OFMulOverflow:%[^ ]*]] = extractvalue { i32, i1 } [[OFMul]], 1
; LV-NEXT: [[AddEnd:%[^ ]*]] = add i32 [[Start:%[^ ]*]], [[OFMulResult]]
; LV-NEXT: [[SubEnd:%[^ ]*]] = sub i32 [[Start]], [[OFMulResult]]
; LV-NEXT: [[CmpNeg:%[^ ]*]] = icmp sgt i32 [[SubEnd]], [[Start]]
; LV-NEXT: [[CmpPos:%[^ ]*]] = icmp slt i32 [[AddEnd]], [[Start]]
; LV-NEXT: [[Cmp:%[^ ]*]] = select i1 true, i1 [[CmpNeg]], i1 [[CmpPos]]
; LV-NEXT: [[BECheck:%[^ ]*]] = icmp ugt i64 [[BE]], 4294967295
; LV-NEXT: [[CheckOr0:%[^ ]*]] = or i1 [[Cmp]], [[BECheck]]
; LV-NEXT: [[PredCheck0:%[^ ]*]] = or i1 [[CheckOr0]], [[OFMulOverflow]]

; LV-NEXT: [[Or0:%[^ ]*]] = or i1 false, [[PredCheck0]]

; LV: [[OFMul1:%[^ ]*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 4, i64 [[BE:%[^ ]*]])
; LV-NEXT: [[OFMulResult1:%[^ ]*]] = extractvalue { i64, i1 } [[OFMul1]], 0
; LV-NEXT: [[OFMulOverflow1:%[^ ]*]] = extractvalue { i64, i1 } [[OFMul1]], 1
; LV-NEXT: [[AddEnd1:%[^ ]*]] = add i64 [[Start:%[^ ]*]], [[OFMulResult1]]
; LV-NEXT: [[SubEnd1:%[^ ]*]] = sub i64 [[Start]], [[OFMulResult1]]
; LV-NEXT: [[CmpNeg1:%[^ ]*]] = icmp ugt i64 [[SubEnd1]], [[Start]]
; LV-NEXT: [[CmpPos1:%[^ ]*]] = icmp ult i64 [[AddEnd1]], [[Start]]
; LV-NEXT: [[Cmp:%[^ ]*]] = select i1 true, i1 [[CmpNeg1]], i1 [[CmpPos1]]
; LV-NEXT: [[PredCheck1:%[^ ]*]] = or i1 [[Cmp]], [[OFMulOverflow1]]

; LV: [[FinalCheck:%[^ ]*]] = or i1 [[Or0]], [[PredCheck1]]
; LV: br i1 [[FinalCheck]], label %for.body.ph.lver.orig, label %for.body.ph
define void @f4(i16* noalias %a,
                i16* noalias %b, i64 %N) {
entry:
  %TruncN = trunc i64 %N to i32
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %ind1 = phi i32 [ %TruncN, %entry ], [ %dec, %for.body ]

  %mul = mul i32 %ind1, 2
  %mul_ext = sext i32 %mul to i64

  %arrayidxA = getelementptr i16, i16* %a, i64 %mul_ext
  %loadA = load i16, i16* %arrayidxA, align 2

  %arrayidxB = getelementptr i16, i16* %b, i64 %ind
  %loadB = load i16, i16* %arrayidxB, align 2

  %add = mul i16 %loadA, %loadB

  store i16 %add, i16* %arrayidxA, align 2

  %inc = add nuw nsw i64 %ind, 1
  %dec = sub i32 %ind1, 1

  %exitcond = icmp eq i64 %inc, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; The following function is similar to the one above, but has the GEP
; to pointer %A inbounds. The index %mul doesn't have the nsw flag.
; This means that the SCEV expression for %mul can wrap and we need
; a SCEV predicate to continue analysis.
;
; We can still analyze this by adding the required no wrap SCEV predicates.

; LAA-LABEL: f5
; LAA: Memory dependences are safe{{$}}
; LAA: SCEV assumptions:
; LAA-NEXT: {(2 * (trunc i64 %N to i32)),+,-2}<%for.body> Added Flags: <nssw>
; LAA-NEXT: {((2 * (sext i32 (2 * (trunc i64 %N to i32)) to i64)) + %a),+,-4}<%for.body> Added Flags: <nusw>

; LAA: [PSE]  %arrayidxA = getelementptr inbounds i16, i16* %a, i32 %mul:
; LAA-NEXT: ((2 * (sext i32 {(2 * (trunc i64 %N to i32)),+,-2}<%for.body> to i64))<nsw> + %a)<nsw>
; LAA-NEXT: --> {((2 * (sext i32 (2 * (trunc i64 %N to i32)) to i64)) + %a),+,-4}<%for.body>

; LV-LABEL: f5
; LV-LABEL: for.body.lver.check
; LV: [[OFMul:%[^ ]*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 2, i32 [[BETrunc:%[^ ]*]])
; LV-NEXT: [[OFMulResult:%[^ ]*]] = extractvalue { i32, i1 } [[OFMul]], 0
; LV-NEXT: [[OFMulOverflow:%[^ ]*]] = extractvalue { i32, i1 } [[OFMul]], 1
; LV-NEXT: [[AddEnd:%[^ ]*]] = add i32 [[Start:%[^ ]*]], [[OFMulResult]]
; LV-NEXT: [[SubEnd:%[^ ]*]] = sub i32 [[Start]], [[OFMulResult]]
; LV-NEXT: [[CmpNeg:%[^ ]*]] = icmp sgt i32 [[SubEnd]], [[Start]]
; LV-NEXT: [[CmpPos:%[^ ]*]] = icmp slt i32 [[AddEnd]], [[Start]]
; LV-NEXT: [[Cmp:%[^ ]*]] = select i1 true, i1 [[CmpNeg]], i1 [[CmpPos]]
; LV-NEXT: [[BECheck:%[^ ]*]] = icmp ugt i64 [[BE]], 4294967295
; LV-NEXT: [[CheckOr0:%[^ ]*]] = or i1 [[Cmp]], [[BECheck]]
; LV-NEXT: [[PredCheck0:%[^ ]*]] = or i1 [[CheckOr0]], [[OFMulOverflow]]

; LV-NEXT: [[Or0:%[^ ]*]] = or i1 false, [[PredCheck0]]

; LV: [[OFMul1:%[^ ]*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 4, i64 [[BE:%[^ ]*]])
; LV-NEXT: [[OFMulResult1:%[^ ]*]] = extractvalue { i64, i1 } [[OFMul1]], 0
; LV-NEXT: [[OFMulOverflow1:%[^ ]*]] = extractvalue { i64, i1 } [[OFMul1]], 1
; LV-NEXT: [[AddEnd1:%[^ ]*]] = add i64 [[Start:%[^ ]*]], [[OFMulResult1]]
; LV-NEXT: [[SubEnd1:%[^ ]*]] = sub i64 [[Start]], [[OFMulResult1]]
; LV-NEXT: [[CmpNeg1:%[^ ]*]] = icmp ugt i64 [[SubEnd1]], [[Start]]
; LV-NEXT: [[CmpPos1:%[^ ]*]] = icmp ult i64 [[AddEnd1]], [[Start]]
; LV-NEXT: [[Cmp:%[^ ]*]] = select i1 true, i1 [[CmpNeg1]], i1 [[CmpPos1]]
; LV-NEXT: [[PredCheck1:%[^ ]*]] = or i1 [[Cmp]], [[OFMulOverflow1]]

; LV: [[FinalCheck:%[^ ]*]] = or i1 [[Or0]], [[PredCheck1]]
; LV: br i1 [[FinalCheck]], label %for.body.ph.lver.orig, label %for.body.ph
define void @f5(i16* noalias %a,
                i16* noalias %b, i64 %N) {
entry:
  %TruncN = trunc i64 %N to i32
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %ind1 = phi i32 [ %TruncN, %entry ], [ %dec, %for.body ]

  %mul = mul i32 %ind1, 2

  %arrayidxA = getelementptr inbounds i16, i16* %a, i32 %mul
  %loadA = load i16, i16* %arrayidxA, align 2

  %arrayidxB = getelementptr inbounds i16, i16* %b, i64 %ind
  %loadB = load i16, i16* %arrayidxB, align 2

  %add = mul i16 %loadA, %loadB

  store i16 %add, i16* %arrayidxA, align 2

  %inc = add nuw nsw i64 %ind, 1
  %dec = sub i32 %ind1, 1

  %exitcond = icmp eq i64 %inc, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
