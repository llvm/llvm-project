; Test that the PutsCatOptimizer works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | \
; RUN:   not grep {call.*fputs}
;

	%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i32, [52 x i8] }
	%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
@stdout = external global %struct._IO_FILE*		; <%struct._IO_FILE**> [#uses=1]
@empty = constant [1 x i8] zeroinitializer		; <[1 x i8]*> [#uses=1]
@len1 = constant [2 x i8] c"A\00"		; <[2 x i8]*> [#uses=1]
@long = constant [7 x i8] c"hello\0A\00"		; <[7 x i8]*> [#uses=1]

declare i32 @fputs(i8*, %struct._IO_FILE*)

define i32 @main() {
entry:
	%out = load %struct._IO_FILE** @stdout		; <%struct._IO_FILE*> [#uses=3]
	%s1 = getelementptr [1 x i8]* @empty, i32 0, i32 0		; <i8*> [#uses=1]
	%s2 = getelementptr [2 x i8]* @len1, i32 0, i32 0		; <i8*> [#uses=1]
	%s3 = getelementptr [7 x i8]* @long, i32 0, i32 0		; <i8*> [#uses=1]
	%a = call i32 @fputs( i8* %s1, %struct._IO_FILE* %out )		; <i32> [#uses=0]
	%b = call i32 @fputs( i8* %s2, %struct._IO_FILE* %out )		; <i32> [#uses=0]
	%c = call i32 @fputs( i8* %s3, %struct._IO_FILE* %out )		; <i32> [#uses=0]
	ret i32 0
}

