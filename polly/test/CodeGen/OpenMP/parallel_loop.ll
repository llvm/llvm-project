; RUN: opt %loadPolly %defaultOpts -polly-cloog -polly-codegen -enable-polly-openmp -analyze  < %s | FileCheck %s
; RUN: opt %loadPolly %defaultOpts -polly-import-jscop -polly-import-jscop-dir=%S -polly-cloog -polly-codegen -enable-polly-openmp -analyze  < %s | FileCheck -check-prefix=IMPORT %s
; RUN: opt %loadPolly %defaultOpts -polly-import-jscop -polly-import-jscop-dir=%S -polly-cloog -polly-codegen -enable-polly-openmp -analyze  < %s | FileCheck -check-prefix=IMPORT %s
; RUN: opt %loadPolly %defaultOpts -polly-import-jscop -polly-import-jscop-postfix=tiled -polly-import-jscop-dir=%S -polly-cloog -polly-codegen -enable-polly-openmp -analyze -disable-polly-legality < %s | FileCheck -check-prefix=TILED %s

;#define M 1024
;#define N 1024
;#define K 1024
;
;float A[M][K], B[K][N], C[M][N], X[K];
;
;float parallel_loop() {
;  int i, j, k;
;
;  for (i = 0; i < M; i++)
;    for (j = 0; j< N; j++)
;      for (k = 0; k < K; k++)
;        C[i][j] += A[i][k] * B[k][j];
;
;  for (i = 0; i < M; i++)
;    for (j = 0; j < N; j++)
;      for (k = 0; k < K; k++)
;        X[k] += X[k];
;
;  return C[42][42] + X[42];
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [1024 x [1024 x float]] zeroinitializer, align 16
@B = common global [1024 x [1024 x float]] zeroinitializer, align 16
@C = common global [1024 x [1024 x float]] zeroinitializer, align 16
@X = common global [1024 x float] zeroinitializer, align 16

define float @parallel_loop() nounwind {
bb:
  br label %bb18

bb18:                                             ; preds = %bb32, %bb
  %indvar9 = phi i64 [ %indvar.next10, %bb32 ], [ 0, %bb ]
  %exitcond15 = icmp ne i64 %indvar9, 1024
  br i1 %exitcond15, label %bb19, label %bb33

bb19:                                             ; preds = %bb18
  br label %bb20

bb20:                                             ; preds = %bb30, %bb19
  %indvar6 = phi i64 [ %indvar.next7, %bb30 ], [ 0, %bb19 ]
  %scevgep14 = getelementptr [1024 x [1024 x float]]* @C, i64 0, i64 %indvar9, i64 %indvar6
  %exitcond12 = icmp ne i64 %indvar6, 1024
  br i1 %exitcond12, label %bb21, label %bb31

bb21:                                             ; preds = %bb20
  br label %bb22

bb22:                                             ; preds = %bb28, %bb21
  %indvar3 = phi i64 [ %indvar.next4, %bb28 ], [ 0, %bb21 ]
  %scevgep11 = getelementptr [1024 x [1024 x float]]* @A, i64 0, i64 %indvar9, i64 %indvar3
  %scevgep8 = getelementptr [1024 x [1024 x float]]* @B, i64 0, i64 %indvar3, i64 %indvar6
  %exitcond5 = icmp ne i64 %indvar3, 1024
  br i1 %exitcond5, label %bb23, label %bb29

bb23:                                             ; preds = %bb22
  %tmp = load float* %scevgep11, align 4
  %tmp24 = load float* %scevgep8, align 4
  %tmp25 = fmul float %tmp, %tmp24
  %tmp26 = load float* %scevgep14, align 4
  %tmp27 = fadd float %tmp26, %tmp25
  store float %tmp27, float* %scevgep14, align 4
  br label %bb28

bb28:                                             ; preds = %bb23
  %indvar.next4 = add i64 %indvar3, 1
  br label %bb22

bb29:                                             ; preds = %bb22
  br label %bb30

bb30:                                             ; preds = %bb29
  %indvar.next7 = add i64 %indvar6, 1
  br label %bb20

bb31:                                             ; preds = %bb20
  br label %bb32

bb32:                                             ; preds = %bb31
  %indvar.next10 = add i64 %indvar9, 1
  br label %bb18

bb33:                                             ; preds = %bb18
  br label %bb34

bb34:                                             ; preds = %bb48, %bb33
  %i.1 = phi i32 [ 0, %bb33 ], [ %tmp49, %bb48 ]
  %exitcond2 = icmp ne i32 %i.1, 1024
  br i1 %exitcond2, label %bb35, label %bb50

bb35:                                             ; preds = %bb34
  br label %bb36

bb36:                                             ; preds = %bb45, %bb35
  %j.1 = phi i32 [ 0, %bb35 ], [ %tmp46, %bb45 ]
  %exitcond1 = icmp ne i32 %j.1, 1024
  br i1 %exitcond1, label %bb37, label %bb47

bb37:                                             ; preds = %bb36
  br label %bb38

bb38:                                             ; preds = %bb43, %bb37
  %indvar = phi i64 [ %indvar.next, %bb43 ], [ 0, %bb37 ]
  %scevgep = getelementptr [1024 x float]* @X, i64 0, i64 %indvar
  %exitcond = icmp ne i64 %indvar, 1024
  br i1 %exitcond, label %bb39, label %bb44

bb39:                                             ; preds = %bb38
  %tmp40 = load float* %scevgep, align 4
  %tmp41 = load float* %scevgep, align 4
  %tmp42 = fadd float %tmp41, %tmp40
  store float %tmp42, float* %scevgep, align 4
  br label %bb43

bb43:                                             ; preds = %bb39
  %indvar.next = add i64 %indvar, 1
  br label %bb38

bb44:                                             ; preds = %bb38
  br label %bb45

bb45:                                             ; preds = %bb44
  %tmp46 = add nsw i32 %j.1, 1
  br label %bb36

bb47:                                             ; preds = %bb36
  br label %bb48

bb48:                                             ; preds = %bb47
  %tmp49 = add nsw i32 %i.1, 1
  br label %bb34

bb50:                                             ; preds = %bb34
  %tmp51 = load float* getelementptr inbounds ([1024 x [1024 x float]]* @C, i64 0, i64 42, i64 42), align 8
  %tmp52 = load float* getelementptr inbounds ([1024 x float]* @X, i64 0, i64 42), align 8
  %tmp53 = fadd float %tmp51, %tmp52
  ret float %tmp53
}

; CHECK: for (c2=0;c2<=1023;c2++) {
; CHECK:   for (c4=0;c4<=1023;c4++) {
; CHECK:     for (c6=0;c6<=1023;c6++) {
; CHECK:       Stmt_bb23(c2,c4,c6);
; CHECK:     }
; CHECK:   }
; CHECK: }
; CHECK: for (c2=0;c2<=1023;c2++) {
; CHECK:   for (c4=0;c4<=1023;c4++) {
; CHECK:     for (c6=0;c6<=1023;c6++) {
; CHECK:       Stmt_bb39(c2,c4,c6);
; CHECK:     }
; CHECK:   }
; CHECK: }
; CHECK: Parallel loop with iterator 'c2' generated
; CHECK: Parallel loop with iterator 'c6' generated
; CHECK-NOT: Parallel loop


; IMPORT: for (c2=0;c2<=1023;c2++) {
; IMPORT:   for (c4=0;c4<=1023;c4++) {
; IMPORT:     for (c6=0;c6<=1023;c6++) {
; IMPORT:       Stmt_bb23(c2,c4,c6);
; IMPORT:       Stmt_bb39(c2,c4,c6);
; IMPORT:     }
; IMPORT:   }
; IMPORT: }
; IMPORT-NOT: Parallel loop

; TILED: for (c2=0;c2<=1023;c2+=4) {
; TILED:   for (c4=0;c4<=1023;c4+=4) {
; TILED:     for (c6=0;c6<=1023;c6+=4) {
; TILED:       for (c8=c2;c8<=c2+3;c8++) {
; TILED:         for (c9=c4;c9<=c4+3;c9++) {
; TILED:           for (c10=c6;c10<=c6+3;c10++) {
; TILED:             Stmt_bb23(c8,c9,c10);
; TILED:           }
; TILED:         }
; TILED:       }
; TILED:     }
; TILED:   }
; TILED: }
; TILED: for (c2=0;c2<=1023;c2+=4) {
; TILED:   for (c4=0;c4<=1023;c4+=4) {
; TILED:     for (c6=0;c6<=1023;c6+=4) {
; TILED:       for (c8=c2;c8<=c2+3;c8++) {
; TILED:         for (c9=c4;c9<=c4+3;c9++) {
; TILED:           for (c10=c6;c10<=c6+3;c10++) {
; TILED:             Stmt_bb39(c8,c9,c10);
; TILED:           }
; TILED:         }
; TILED:       }
; TILED:     }
; TILED:   }
; TILED: }
; I am not sure if we actually may have parallel loops here. The dependency
; analysis does not detect any. This may however be because we do not
; correctly update the imported schedule. Add a check that hopefully fails
; after this is corrected. Or someone proves there are no parallel loops and
; we can remove this comment.
; TILDED-NOT: Parallel loop
