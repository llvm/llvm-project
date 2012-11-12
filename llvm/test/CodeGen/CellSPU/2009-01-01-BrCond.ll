; RUN: llc < %s -march=cellspu -o - | grep brz
; PR3274

target datalayout = "E-p:32:32:128-i1:8:128-i8:8:128-i16:16:128-i32:32:128-i64:32:128-f32:32:128-f64:64:128-v64:64:64-v128:128:128-a0:0:128-s0:128:128"
target triple = "spu"
	%struct.anon = type { i64 }
	%struct.fp_number_type = type { i32, i32, i32, [4 x i8], %struct.anon }

define double @__floatunsidf(i32 %arg_a) nounwind {
entry:
	%in = alloca %struct.fp_number_type, align 16
	%0 = getelementptr %struct.fp_number_type* %in, i32 0, i32 1
	store i32 0, i32* %0, align 4
	%1 = icmp eq i32 %arg_a, 0
	%2 = getelementptr %struct.fp_number_type* %in, i32 0, i32 0
	br i1 %1, label %bb, label %bb1

bb:		; preds = %entry
	store i32 2, i32* %2, align 8
	br label %bb7

bb1:		; preds = %entry
	ret double 0.0

bb7:		; preds = %bb5, %bb1, %bb
	ret double 1.0
}

; declare i32 @llvm.ctlz.i32(i32) nounwind readnone

declare double @__pack_d(%struct.fp_number_type*)
