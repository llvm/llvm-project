; RUN: llc < %s -mtriple=thumbv7-none-linux-gnueabi
; PR4681

	%struct.FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct.FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i32, i32, [40 x i8] }
	%struct._IO_marker = type { %struct._IO_marker*, %struct.FILE*, i32 }
@.str2 = external constant [30 x i8], align 1		; <[30 x i8]*> [#uses=1]

define i32 @__mf_heuristic_check(i32 %ptr, i32 %ptr_high) nounwind {
entry:
	br i1 undef, label %bb1, label %bb

bb:		; preds = %entry
	unreachable

bb1:		; preds = %entry
	br i1 undef, label %bb9, label %bb2

bb2:		; preds = %bb1
	%0 = call i8* @llvm.frameaddress(i32 0)		; <i8*> [#uses=1]
	%1 = call  i32 (%struct.FILE*, i8*, ...)* @fprintf(%struct.FILE* noalias undef, i8* noalias getelementptr ([30 x i8]* @.str2, i32 0, i32 0), i8* %0, i8* null) nounwind		; <i32> [#uses=0]
	unreachable

bb9:		; preds = %bb1
	ret i32 undef
}

declare i8* @llvm.frameaddress(i32) nounwind readnone

declare i32 @fprintf(%struct.FILE* noalias nocapture, i8* noalias nocapture, ...) nounwind
