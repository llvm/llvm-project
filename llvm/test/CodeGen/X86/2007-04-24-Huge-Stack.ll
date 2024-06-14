; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s
; PR1348

; CHECK-NOT: 4294967112

	%struct.md5_ctx = type { i32, i32, i32, i32, [2 x i32], i32, [128 x i8], [4294967288 x i8] }

define ptr @md5_buffer(ptr %buffer, i64 %len, ptr %resblock) {
entry:
	%ctx = alloca %struct.md5_ctx, align 16		; <ptr> [#uses=3]
	call void @md5_init_ctx( ptr %ctx )
	call void @md5_process_bytes( ptr %buffer, i64 %len, ptr %ctx )
	%tmp4 = call ptr @md5_finish_ctx( ptr %ctx, ptr %resblock )		; <ptr> [#uses=1]
	ret ptr %tmp4
}

declare void @md5_init_ctx(ptr)

declare ptr @md5_finish_ctx(ptr, ptr)

declare void @md5_process_bytes(ptr, i64, ptr)
