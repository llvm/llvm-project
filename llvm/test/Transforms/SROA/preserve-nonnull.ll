; RUN: opt < %s -sroa -S | FileCheck %s
;
; Make sure that SROA doesn't lose nonnull metadata
; on loads from allocas that get optimized out.

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1)

; Check that we do basic propagation of nonnull when rewriting.
define i8* @propagate_nonnull(i32* %v) {
; CHECK-LABEL: define i8* @propagate_nonnull(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %[[A:.*]] = alloca i8*
; CHECK-NEXT:    %[[V_CAST:.*]] = bitcast i32* %v to i8*
; CHECK-NEXT:    store i8* %[[V_CAST]], i8** %[[A]]
; CHECK-NEXT:    %[[LOAD:.*]] = load volatile i8*, i8** %[[A]], !nonnull !0
; CHECK-NEXT:    ret i8* %[[LOAD]]
entry:
  %a = alloca [2 x i8*]
  %a.gep0 = getelementptr [2 x i8*], [2 x i8*]* %a, i32 0, i32 0
  %a.gep1 = getelementptr [2 x i8*], [2 x i8*]* %a, i32 0, i32 1
  %a.gep0.cast = bitcast i8** %a.gep0 to i32**
  %a.gep1.cast = bitcast i8** %a.gep1 to i32**
  store i32* %v, i32** %a.gep1.cast
  store i32* null, i32** %a.gep0.cast
  %load = load volatile i8*, i8** %a.gep1, !nonnull !0
  ret i8* %load
}

define float* @turn_nonnull_into_assume(float** %arg) {
; CHECK-LABEL: define float* @turn_nonnull_into_assume(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %[[RETURN:.*]] = load float*, float** %arg, align 8
; CHECK-NEXT:    %[[ASSUME:.*]] = icmp ne float* %[[RETURN]], null
; CHECK-NEXT:    call void @llvm.assume(i1 %[[ASSUME]])
; CHECK-NEXT:    ret float* %[[RETURN]]
entry:
  %buf = alloca float*
  %_arg_i8 = bitcast float** %arg to i8*
  %_buf_i8 = bitcast float** %buf to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %_buf_i8, i8* %_arg_i8, i64 8, i32 8, i1 false)
  %ret = load float*, float** %buf, align 8, !nonnull !0
  ret float* %ret
}

!0 = !{}
