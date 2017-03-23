; RUN: opt < %s -sroa -S | FileCheck %s
;
; Make sure that SROA doesn't lose nonnull metadata
; on loads from allocas that get optimized out.

; CHECK-LABEL: define float* @yummy_nonnull
; CHECK: [[RETURN:%(.*)]] = load float*, float** %arg, align 8
; CHECK: [[ASSUME:%(.*)]] = icmp ne float* {{.*}}[[RETURN]], null
; CHECK: call void @llvm.assume(i1 {{.*}}[[ASSUME]])
; CHECK: ret float* {{.*}}[[RETURN]]

define float* @yummy_nonnull(float** %arg) {
entry-block:
	%buf = alloca float*

	%_arg_i8 = bitcast float** %arg to i8*
	%_buf_i8 = bitcast float** %buf to i8*
	call void @llvm.memcpy.p0i8.p0i8.i64(i8* %_buf_i8, i8* %_arg_i8, i64 8, i32 8, i1 false)

	%ret = load float*, float** %buf, align 8, !nonnull !0
	ret float* %ret
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1)

!0 = !{}
