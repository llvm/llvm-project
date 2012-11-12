; Test that the FPrintFOptimizer works correctly
; RUN: opt < %s -simplify-libcalls -S | \
; RUN:   not grep "call.*fprintf"

; This transformation requires the pointer size, as it assumes that size_t is
; the size of a pointer.
target datalayout = "-p:64:64:64"

	%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i32, [52 x i8] }
	%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
@str = constant [3 x i8] c"%s\00"		; <[3 x i8]*> [#uses=1]
@chr = constant [3 x i8] c"%c\00"		; <[3 x i8]*> [#uses=1]
@hello = constant [13 x i8] c"hello world\0A\00"		; <[13 x i8]*> [#uses=1]
@stdout = external global %struct._IO_FILE*		; <%struct._IO_FILE**> [#uses=3]

declare i32 @fprintf(%struct._IO_FILE*, i8*, ...)

define i32 @foo() {
entry:
	%tmp.1 = load %struct._IO_FILE** @stdout		; <%struct._IO_FILE*> [#uses=1]
	%tmp.0 = call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf( %struct._IO_FILE* %tmp.1, i8* getelementptr ([13 x i8]* @hello, i32 0, i32 0) )		; <i32> [#uses=0]
	%tmp.4 = load %struct._IO_FILE** @stdout		; <%struct._IO_FILE*> [#uses=1]
	%tmp.3 = call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf( %struct._IO_FILE* %tmp.4, i8* getelementptr ([3 x i8]* @str, i32 0, i32 0), i8* getelementptr ([13 x i8]* @hello, i32 0, i32 0) )		; <i32> [#uses=0]
	%tmp.8 = load %struct._IO_FILE** @stdout		; <%struct._IO_FILE*> [#uses=1]
	%tmp.7 = call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf( %struct._IO_FILE* %tmp.8, i8* getelementptr ([3 x i8]* @chr, i32 0, i32 0), i32 33 )		; <i32> [#uses=0]
	ret i32 0
}

