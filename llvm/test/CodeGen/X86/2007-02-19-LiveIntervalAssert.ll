; RUN: llc < %s -march=x86 -mtriple=i686-pc-linux-gnu -relocation-model=pic
; PR1027

	%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i32, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i32, i32, [40 x i8] }
	%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
@stderr = external global %struct._IO_FILE*

define void @__eprintf(i8* %string, i8* %expression, i32 %line, i8* %filename) {
	%tmp = load %struct._IO_FILE** @stderr
	%tmp5 = tail call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf( %struct._IO_FILE* %tmp, i8* %string, i8* %expression, i32 %line, i8* %filename )
	%tmp6 = load %struct._IO_FILE** @stderr
	%tmp7 = tail call i32 @fflush( %struct._IO_FILE* %tmp6 )
	tail call void @abort( )
	unreachable
}

declare i32 @fprintf(%struct._IO_FILE*, i8*, ...)

declare i32 @fflush(%struct._IO_FILE*)

declare void @abort()
