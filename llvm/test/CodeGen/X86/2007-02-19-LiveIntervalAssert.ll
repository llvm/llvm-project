; RUN: llc < %s -mtriple=i686-pc-linux-gnu -relocation-model=pic
; PR1027

	%struct._IO_FILE = type { i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i16, i8, [1 x i8], ptr, i64, ptr, ptr, ptr, ptr, i32, i32, [40 x i8] }
	%struct._IO_marker = type { ptr, ptr, i32 }
@stderr = external global ptr

define void @__eprintf(ptr %string, ptr %expression, i32 %line, ptr %filename) {
	%tmp = load ptr, ptr @stderr
	%tmp5 = tail call i32 (ptr, ptr, ...) @fprintf( ptr %tmp, ptr %string, ptr %expression, i32 %line, ptr %filename )
	%tmp6 = load ptr, ptr @stderr
	%tmp7 = tail call i32 @fflush( ptr %tmp6 )
	tail call void @abort( )
	unreachable
}

declare i32 @fprintf(ptr, ptr, ...)

declare i32 @fflush(ptr)

declare void @abort()
