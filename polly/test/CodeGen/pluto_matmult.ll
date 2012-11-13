; RUN: opt %loadPolly %defaultOpts -polly-cloog -analyze < %s | FileCheck %s
; RUN: opt %loadPolly %defaultOpts -polly-codegen < %s > /dev/null
; RUN: opt %loadPolly %defaultOpts -polly-import-jscop -polly-import-jscop-dir=%S -polly-cloog -analyze  -S < %s | FileCheck -check-prefix=IMPORT %s

;#define M 2048
;#define N 2048
;#define K 2048
;#define alpha 1
;#define beta 1
;double A[M][K+13];
;double B[K][N+13];
;double C[M][N+13];
;
;void init_array();
;void print_array();
;
;void pluto_matmult(void) {
;  int i, j, k;
;
;  __sync_synchronize();
;  for(i=0; i<M; i++)
;    for(j=0; j<N; j++)
;      for(k=0; k<K; k++)
;        C[i][j] = beta*C[i][j] + alpha*A[i][k] * B[k][j];
;  __sync_synchronize();
;}
;
;int main()
;{
;    register double s;
;
;    init_array();
;
;#pragma scop
;    pluto_matmult();
;#pragma endscop
;    print_array();
;
;  return 0;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@C = common global [2048 x [2061 x double]] zeroinitializer, align 8 ; <[2048 x [2061 x double]]*> [#uses=2]
@A = common global [2048 x [2061 x double]] zeroinitializer, align 8 ; <[2048 x [2061 x double]]*> [#uses=2]
@B = common global [2048 x [2061 x double]] zeroinitializer, align 8 ; <[2048 x [2061 x double]]*> [#uses=2]

define void @pluto_matmult() nounwind {
entry:
  fence seq_cst
  br label %for.cond

for.cond:                                         ; preds = %for.inc44, %entry
  %indvar3 = phi i64 [ %indvar.next4, %for.inc44 ], [ 0, %entry ] ; <i64> [#uses=4]
  %exitcond6 = icmp ne i64 %indvar3, 2048         ; <i1> [#uses=1]
  br i1 %exitcond6, label %for.body, label %for.end47

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc40, %for.body
  %indvar1 = phi i64 [ %indvar.next2, %for.inc40 ], [ 0, %for.body ] ; <i64> [#uses=4]
  %arrayidx12 = getelementptr [2048 x [2061 x double]]* @C, i64 0, i64 %indvar3, i64 %indvar1 ; <double*> [#uses=2]
  %exitcond5 = icmp ne i64 %indvar1, 2048         ; <i1> [#uses=1]
  br i1 %exitcond5, label %for.body4, label %for.end43

for.body4:                                        ; preds = %for.cond1
  br label %for.cond5

for.cond5:                                        ; preds = %for.inc, %for.body4
  %indvar = phi i64 [ %indvar.next, %for.inc ], [ 0, %for.body4 ] ; <i64> [#uses=4]
  %arrayidx20 = getelementptr [2048 x [2061 x double]]* @A, i64 0, i64 %indvar3, i64 %indvar ; <double*> [#uses=1]
  %arrayidx29 = getelementptr [2048 x [2061 x double]]* @B, i64 0, i64 %indvar, i64 %indvar1 ; <double*> [#uses=1]
  %exitcond = icmp ne i64 %indvar, 2048           ; <i1> [#uses=1]
  br i1 %exitcond, label %for.body8, label %for.end

for.body8:                                        ; preds = %for.cond5
  %tmp13 = load double* %arrayidx12               ; <double> [#uses=1]
  %mul = fmul double 1.000000e+00, %tmp13         ; <double> [#uses=1]
  %tmp21 = load double* %arrayidx20               ; <double> [#uses=1]
  %mul22 = fmul double 1.000000e+00, %tmp21       ; <double> [#uses=1]
  %tmp30 = load double* %arrayidx29               ; <double> [#uses=1]
  %mul31 = fmul double %mul22, %tmp30             ; <double> [#uses=1]
  %add = fadd double %mul, %mul31                 ; <double> [#uses=1]
  store double %add, double* %arrayidx12
  br label %for.inc

for.inc:                                          ; preds = %for.body8
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br label %for.cond5

for.end:                                          ; preds = %for.cond5
  br label %for.inc40

for.inc40:                                        ; preds = %for.end
  %indvar.next2 = add i64 %indvar1, 1             ; <i64> [#uses=1]
  br label %for.cond1

for.end43:                                        ; preds = %for.cond1
  br label %for.inc44

for.inc44:                                        ; preds = %for.end43
  %indvar.next4 = add i64 %indvar3, 1             ; <i64> [#uses=1]
  br label %for.cond

for.end47:                                        ; preds = %for.cond
  fence seq_cst
  ret void
}

define i32 @main() nounwind {
entry:
  call void (...)* @init_array()
  call void @pluto_matmult()
  call void (...)* @print_array()
  ret i32 0
}

declare void @init_array(...)

declare void @print_array(...)
; CHECK:  for (c2=0;c2<=2047;c2++) {
; CHECK:    for (c4=0;c4<=2047;c4++) {
; CHECK:      for (c6=0;c6<=2047;c6++) {
; CHECK:        Stmt_for_body8(c2,c4,c6);
; CHECK:      }
; CHECK:    }
; CHECK:  }


; Do not dump the complete CLooG output. New CLooG version optimize more
; in this test case.
; IMPORT: for (c2=0;c2<=2047;c2+=64) {
; IMPORT:   c3<=min(2047,c2+63);c3++) {
; IMPORT:     for (c6=0;c6<=2047;c6+=64) {
; IMPORT:       c7<=min(2047,c6+63);c7++) {
; IMPORT:         for (c10=0;c10<=2047;c10+=64) {
; IMPORT:           c11<=min(2047,c10+63);c11++)
; IMPORT:             {
; IMPORT:               Stmt_for_body8(c3,c7,c11);
; IMPORT:             }
; IMPORT:         }
; IMPORT:       }
; IMPORT:     }
; IMPORT:   }
; IMPORT: }

