; RUN: opt %loadPolly -basic-aa -polly-codegen -polly-vectorizer=polly -S \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@A = common global [1024 x float**] zeroinitializer, align 16
@B = common global [1024 x float**] zeroinitializer, align 16

declare float @foo(float) readnone

define void @simple_vec_call() nounwind {
entry:
  br label %body

body:
  %indvar = phi i64 [ 0, %entry ], [ %indvar_next, %body ]
  %scevgep = getelementptr [1024 x float**], [1024 x float**]* @B, i64 0, i64 %indvar
  %value = load float**, float*** getelementptr inbounds ([1024 x float**], [1024 x float**]* @A, i64 0, i64 0), align 16
  store float** %value, float*** %scevgep, align 4
  %indvar_next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar_next, 4
  br i1 %exitcond, label %return, label %body

return:
  ret void
}
; CHECK:   %.load = load float**, float*** getelementptr inbounds ([1024 x float**], [1024 x float**]* @A, i32 0, i32 0)

; CHECK-NOT: load <1 x float**>
; CHECK: %value_p.splatinsert = insertelement <4 x float**> poison, float** %.load, i32 0
; CHECK: %value_p.splat = shufflevector <4 x float**> %value_p.splatinsert, <4 x float**> poison, <4 x i32> zeroinitializer
; CHECK: store <4 x float**> %value_p.splat, <4 x float**>* bitcast ([1024 x float**]* @B to <4 x float**>*), align 8
