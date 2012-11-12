; RUN: opt < %s -simplify-libcalls -disable-output
; PR4641

	%struct.__sFILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, i8*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64, %struct.pthread_mutex*, %struct.pthread*, i32, i32, %union.anon }
	%struct.__sbuf = type { i8*, i32, [4 x i8] }
	%struct.pthread = type opaque
	%struct.pthread_mutex = type opaque
	%union.anon = type { i64, [120 x i8] }
@.str13 = external constant [2 x i8]		; <[2 x i8]*> [#uses=1]
@.str14 = external constant [2 x i8]		; <[2 x i8]*> [#uses=1]

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
	call void @exit(i32 0) nounwind
	%cond392 = select i1 undef, i8* getelementptr ([2 x i8]* @.str13, i32 0, i32 0), i8* getelementptr ([2 x i8]* @.str14, i32 0, i32 0)		; <i8*> [#uses=1]
	%call393 = call %struct.__sFILE* @fopen(i8* undef, i8* %cond392) nounwind		; <%struct.__sFILE*> [#uses=0]
	unreachable
}

declare %struct.__sFILE* @fopen(i8*, i8*)

declare void @exit(i32)
