; RUN: llc < %s -mtriple=arm-apple-darwin -mattr=+v6 -relocation-model=pic | grep comm

	%struct.FILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
	%struct.__gcov_var = type { %struct.FILE*, i32, i32, i32, i32, i32, i32, [1025 x i32] }
	%struct.__sFILEX = type opaque
	%struct.__sbuf = type { i8*, i32 }
@__gcov_var = common global %struct.__gcov_var zeroinitializer		; <%struct.__gcov_var*> [#uses=1]

define i32 @__gcov_close() nounwind {
entry:
	load i32* getelementptr (%struct.__gcov_var* @__gcov_var, i32 0, i32 5), align 4		; <i32>:0 [#uses=1]
	ret i32 %0
}
