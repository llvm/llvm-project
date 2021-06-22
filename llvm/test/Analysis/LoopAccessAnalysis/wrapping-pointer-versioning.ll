; RUN: opt -basic-aa -loop-accesses -analyze -enable-new-pm=0 < %s | FileCheck %s -check-prefix=LAA
; RUN: opt -passes='require<aa>,require<scalar-evolution>,require<aa>,loop(print-access-info)' -aa-pipeline='basic-aa' -disable-output < %s  2>&1 | FileCheck %s --check-prefix=LAA

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
; LAA-NEXT: ((2 * (zext i32 {0,+,2}<%for.body> to i64))<nuw><nsw> + %a)
; LAA-NEXT: --> {%a,+,4}<%for.body>


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
; LAA-NEXT: {((4 * (zext i31 (trunc i64 %N to i31) to i64))<nuw><nsw> + %a),+,-4}<%for.body> Added Flags: <nusw>

; The expression for %mul_ext as analyzed by SCEV is
;     (zext i32 {(2 * (trunc i64 %N to i32)),+,-2}<%for.body> to i64)
; We have added the nusw flag to turn this expression into the following SCEV:
;     i64 {zext i32 (2 * (trunc i64 %N to i32)) to i64,+,-2}<%for.body>

; LAA: [PSE]  %arrayidxA = getelementptr i16, i16* %a, i64 %mul_ext:
; LAA-NEXT: ((2 * (zext i32 {(2 * (trunc i64 %N to i32)),+,-2}<%for.body> to i64))<nuw><nsw> + %a)
; LAA-NEXT: --> {((4 * (zext i31 (trunc i64 %N to i31) to i64))<nuw><nsw> + %a),+,-4}<%for.body>

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
; LAA-NEXT: ((2 * (sext i32 {0,+,2}<%for.body> to i64))<nsw> + %a)
; LAA-NEXT: --> {%a,+,4}<%for.body>

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
; LAA-NEXT: {((2 * (sext i32 (2 * (trunc i64 %N to i32)) to i64))<nsw> + %a),+,-4}<%for.body> Added Flags: <nusw>

; The expression for %mul_ext as analyzed by SCEV is
;     i64  (sext i32 {(2 * (trunc i64 %N to i32)),+,-2}<%for.body> to i64)
; We have added the nssw flag to turn this expression into the following SCEV:
;     i64 {sext i32 (2 * (trunc i64 %N to i32)) to i64,+,-2}<%for.body>

; LAA: [PSE]  %arrayidxA = getelementptr i16, i16* %a, i64 %mul_ext:
; LAA-NEXT: ((2 * (sext i32 {(2 * (trunc i64 %N to i32)),+,-2}<%for.body> to i64))<nsw> + %a)
; LAA-NEXT: --> {((2 * (sext i32 (2 * (trunc i64 %N to i32)) to i64))<nsw> + %a),+,-4}<%for.body>

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
; LAA-NEXT: {((2 * (sext i32 (2 * (trunc i64 %N to i32)) to i64))<nsw> + %a),+,-4}<%for.body> Added Flags: <nusw>

; LAA: [PSE]  %arrayidxA = getelementptr inbounds i16, i16* %a, i32 %mul:
; LAA-NEXT: ((2 * (sext i32 {(2 * (trunc i64 %N to i32)),+,-2}<%for.body> to i64))<nsw> + %a)
; LAA-NEXT: --> {((2 * (sext i32 (2 * (trunc i64 %N to i32)) to i64))<nsw> + %a),+,-4}<%for.body>

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
