; RUN: opt %loadPolly -basicaa -polly-dependences -analyze -polly-dependences-analysis-type=value-based < %s | FileCheck %s -check-prefix=VALUE
; RUN: opt %loadPolly -basicaa -polly-dependences -analyze -polly-dependences-analysis-type=memory-based < %s | FileCheck %s -check-prefix=MEMORY
; RUN: opt %loadPolly -basicaa -polly-function-dependences -analyze -polly-dependences-analysis-type=value-based < %s | FileCheck %s -check-prefix=FUNC-VALUE
; RUN: opt %loadPolly -basicaa -polly-function-dependences -analyze -polly-dependences-analysis-type=memory-based < %s | FileCheck %s -check-prefix=FUNC-MEMORY

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@A = common global [36 x [49 x double]] zeroinitializer, align 8 ; <[36 x [49 x double]]*> [#uses=3]
@B = common global [36 x [49 x double]] zeroinitializer, align 8 ; <[36 x [49 x double]]*> [#uses=3]
@C = common global [36 x [49 x double]] zeroinitializer, align 8 ; <[36 x [49 x double]]*> [#uses=4]

define void @do_pluto_matmult() nounwind {
entry:
  fence seq_cst
  br label %do.body

do.body:                                          ; preds = %do.cond42, %entry
  %indvar3 = phi i64 [ %indvar.next4, %do.cond42 ], [ 0, %entry ] ; <i64> [#uses=3]
  br label %do.body1

do.body1:                                         ; preds = %do.cond36, %do.body
  %indvar1 = phi i64 [ %indvar.next2, %do.cond36 ], [ 0, %do.body ] ; <i64> [#uses=3]
  %arrayidx5 = getelementptr [36 x [49 x double]], [36 x [49 x double]]* @C, i64 0, i64 %indvar3, i64 %indvar1 ; <double*> [#uses=2]
  br label %do.body2

do.body2:                                         ; preds = %do.cond, %do.body1
  %indvar = phi i64 [ %indvar.next, %do.cond ], [ 0, %do.body1 ] ; <i64> [#uses=3]
  %arrayidx13 = getelementptr [36 x [49 x double]], [36 x [49 x double]]* @A, i64 0, i64 %indvar3, i64 %indvar ; <double*> [#uses=1]
  %arrayidx22 = getelementptr [36 x [49 x double]], [36 x [49 x double]]* @B, i64 0, i64 %indvar, i64 %indvar1 ; <double*> [#uses=1]
  %tmp6 = load double, double* %arrayidx5                 ; <double> [#uses=1]
  %mul = fmul double 1.000000e+00, %tmp6          ; <double> [#uses=1]
  %tmp14 = load double, double* %arrayidx13               ; <double> [#uses=1]
  %mul15 = fmul double 1.000000e+00, %tmp14       ; <double> [#uses=1]
  %tmp23 = load double, double* %arrayidx22               ; <double> [#uses=1]
  %mul24 = fmul double %mul15, %tmp23             ; <double> [#uses=1]
  %add = fadd double %mul, %mul24                 ; <double> [#uses=1]
  store double %add, double* %arrayidx5
  br label %do.cond

do.cond:                                          ; preds = %do.body2
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
  %exitcond = icmp ne i64 %indvar.next, 36        ; <i1> [#uses=1]
  br i1 %exitcond, label %do.body2, label %do.end

do.end:                                           ; preds = %do.cond
  br label %do.cond36

do.cond36:                                        ; preds = %do.end
  %indvar.next2 = add i64 %indvar1, 1             ; <i64> [#uses=2]
  %exitcond5 = icmp ne i64 %indvar.next2, 36      ; <i1> [#uses=1]
  br i1 %exitcond5, label %do.body1, label %do.end39

do.end39:                                         ; preds = %do.cond36
  br label %do.cond42

do.cond42:                                        ; preds = %do.end39
  %indvar.next4 = add i64 %indvar3, 1             ; <i64> [#uses=2]
  %exitcond6 = icmp ne i64 %indvar.next4, 36      ; <i1> [#uses=1]
  br i1 %exitcond6, label %do.body, label %do.end45

do.end45:                                         ; preds = %do.cond42
  fence seq_cst
  ret void
}

; VALUE:      RAW dependences:
; VALUE-NEXT:     { Stmt_do_body2[i0, i1, i2] -> Stmt_do_body2[i0, i1, 1 + i2] : 0 <= i0 <= 35 and 0 <= i1 <= 35 and 0 <= i2 <= 34 }
; VALUE-NEXT: WAR dependences:
; VALUE-NEXT: { Stmt_do_body2[i0, i1, i2] -> Stmt_do_body2[i0, i1, 1 + i2] : 0 <= i0 <= 35 and 0 <= i1 <= 35 and 0 <= i2 <= 34 }
; VALUE-NEXT: WAW dependences:
; VALUE-NEXT:     { Stmt_do_body2[i0, i1, i2] -> Stmt_do_body2[i0, i1, 1 + i2] : 0 <= i0 <= 35 and 0 <= i1 <= 35 and 0 <= i2 <= 34 }

; MEMORY:      RAW dependences:
; MEMORY-NEXT:     { Stmt_do_body2[i0, i1, i2] -> Stmt_do_body2[i0, i1, o2] : 0 <= i0 <= 35 and 0 <= i1 <= 35 and i2 >= 0 and i2 < o2 <= 35 }
; MEMORY-NEXT: WAR dependences:
; MEMORY-NEXT:     { Stmt_do_body2[i0, i1, i2] -> Stmt_do_body2[i0, i1, o2] : 0 <= i0 <= 35 and 0 <= i1 <= 35 and i2 >= 0 and i2 < o2 <= 35 }
; MEMORY-NEXT: WAW dependences:
; MEMORY-NEXT:     { Stmt_do_body2[i0, i1, i2] -> Stmt_do_body2[i0, i1, o2] : 0 <= i0 <= 35 and 0 <= i1 <= 35 and i2 >= 0 and i2 < o2 <= 35 }

; FUNC-VALUE:      RAW dependences:
; FUNC-VALUE-NEXT:     { [Stmt_do_body2[i0, i1, i2] -> Stmt_do_body2_Write3_MemRef_C[]] -> [Stmt_do_body2[i0, i1, 1 + i2] -> Stmt_do_body2_Read0_MemRef_C[]] : 0 <= i0 <= 35 and 0 <= i1 <= 35 and 0 <= i2 <= 34; Stmt_do_body2[i0, i1, i2] -> Stmt_do_body2[i0, i1, 1 + i2] : 0 <= i0 <= 35 and 0 <= i1 <= 35 and 0 <= i2 <= 34 }
; FUNC-VALUE-NEXT: WAR dependences:
; FUNC-VALUE-NEXT:     { }
; FUNC-VALUE-NEXT: WAW dependences:
; FUNC-VALUE-NEXT:     { [Stmt_do_body2[i0, i1, i2] -> Stmt_do_body2_Write3_MemRef_C[]] -> [Stmt_do_body2[i0, i1, 1 + i2] -> Stmt_do_body2_Write3_MemRef_C[]] : 0 <= i0 <= 35 and 0 <= i1 <= 35 and 0 <= i2 <= 34; Stmt_do_body2[i0, i1, i2] -> Stmt_do_body2[i0, i1, 1 + i2] : 0 <= i0 <= 35 and 0 <= i1 <= 35 and 0 <= i2 <= 34 }

; FUNC-MEMORY:      RAW dependences:
; FUNC-MEMORY-NEXT:     { [Stmt_do_body2[i0, i1, i2] -> Stmt_do_body2_Write3_MemRef_C[]] -> [Stmt_do_body2[i0, i1, o2] -> Stmt_do_body2_Read0_MemRef_C[]] : 0 <= i0 <= 35 and 0 <= i1 <= 35 and i2 >= 0 and i2 < o2 <= 35; Stmt_do_body2[i0, i1, i2] -> Stmt_do_body2[i0, i1, o2] : 0 <= i0 <= 35 and 0 <= i1 <= 35 and i2 >= 0 and i2 < o2 <= 35 }
; FUNC-MEMORY-NEXT: WAR dependences:
; FUNC-MEMORY-NEXT:     { [Stmt_do_body2[i0, i1, i2] -> Stmt_do_body2_Read0_MemRef_C[]] -> [Stmt_do_body2[i0, i1, o2] -> Stmt_do_body2_Write3_MemRef_C[]] : 0 <= i0 <= 35 and 0 <= i1 <= 35 and i2 >= 0 and i2 < o2 <= 35; Stmt_do_body2[i0, i1, i2] -> Stmt_do_body2[i0, i1, o2] : 0 <= i0 <= 35 and 0 <= i1 <= 35 and i2 >= 0 and i2 < o2 <= 35 }
; FUNC-MEMORY-NEXT: WAW dependences:
; FUNC-MEMORY-NEXT:     { [Stmt_do_body2[i0, i1, i2] -> Stmt_do_body2_Write3_MemRef_C[]] -> [Stmt_do_body2[i0, i1, o2] -> Stmt_do_body2_Write3_MemRef_C[]] : 0 <= i0 <= 35 and 0 <= i1 <= 35 and i2 >= 0 and i2 < o2 <= 35; Stmt_do_body2[i0, i1, i2] -> Stmt_do_body2[i0, i1, o2] : 0 <= i0 <= 35 and 0 <= i1 <= 35 and i2 >= 0 and i2 < o2 <= 35 }
