; RUN: llc < %s -march=x86-64 | not grep 4294967112
; PR1348

	%struct.md5_ctx = type { i32, i32, i32, i32, [2 x i32], i32, [128 x i8], [4294967288 x i8] }

define i8* @md5_buffer(i8* %buffer, i64 %len, i8* %resblock) {
entry:
	%ctx = alloca %struct.md5_ctx, align 16		; <%struct.md5_ctx*> [#uses=3]
	call void @md5_init_ctx( %struct.md5_ctx* %ctx )
	call void @md5_process_bytes( i8* %buffer, i64 %len, %struct.md5_ctx* %ctx )
	%tmp4 = call i8* @md5_finish_ctx( %struct.md5_ctx* %ctx, i8* %resblock )		; <i8*> [#uses=1]
	ret i8* %tmp4
}

declare void @md5_init_ctx(%struct.md5_ctx*)

declare i8* @md5_finish_ctx(%struct.md5_ctx*, i8*)

declare void @md5_process_bytes(i8*, i64, %struct.md5_ctx*)
