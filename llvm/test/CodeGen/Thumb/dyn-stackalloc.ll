; RUN: llc < %s -mtriple=thumb-apple-darwin -disable-cgp-branch-opts -disable-post-ra -verify-machineinstrs | FileCheck %s -check-prefix=CHECK -check-prefix=RA_GREEDY
; RUN: llc < %s -mtriple=thumb-apple-darwin -disable-cgp-branch-opts -disable-post-ra -regalloc=basic -verify-machineinstrs | FileCheck %s -check-prefix=CHECK -check-prefix=RA_BASIC

	%struct.state = type { i32, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64, i64, ptr }
	%struct.info = type { i32, i32, i32, i32, i32, i32, i32, ptr }

define void @t1(ptr %v) {
; CHECK-LABEL: t1:
; CHECK: push
; CHECK: add r7, sp, #12
; CHECK: lsls r[[R0:[0-9]+]]
; CHECK: mov r[[R1:[0-9]+]], sp
; CHECK: subs r[[R2:[0-9]+]], r[[R1]], r[[R0]]
; CHECK: mov sp, r[[R2]]
	%tmp6 = load i32, ptr null
	%tmp8 = alloca float, i32 %tmp6
	store i32 1, ptr null
	br i1 false, label %bb123.preheader, label %return

bb123.preheader:
	br i1 false, label %bb43, label %return

bb43:
	call fastcc void @f1( ptr %tmp8, ptr null, i32 0 )
	%tmp70 = load i32, ptr null
	call fastcc void @f2( ptr null, ptr null, ptr %tmp8, i32 %tmp70 )
	ret void

return:
	ret void
}

declare fastcc void @f1(ptr, ptr, i32)

declare fastcc void @f2(ptr, ptr, ptr, i32)

	%struct.comment = type { ptr, ptr, i32, ptr }
@str215 = external global [2 x i8]

define void @t2(ptr %vc, ptr %tag, ptr %contents) {
; CHECK-LABEL: t2:
; CHECK: push
; CHECK: add r7, sp, #12
; CHECK: sub sp, #
; CHECK: mov r[[R0:[0-9]+]], sp
; CHECK: str r{{[0-9+]}}, [r[[R0]]
; RA_GREEDY: str r{{[0-9+]}}, [r[[R0]]
; RA_BASIC: stm r[[R0]]!
; CHECK-NOT: ldr r0, [sp
; CHECK: mov r[[R1:[0-9]+]], sp
; CHECK: subs r[[R2:[0-9]+]], r[[R1]], r{{[0-9]+}}
; CHECK: mov sp, r[[R2]]
; CHECK-NOT: ldr r0, [sp
; CHECK: bx
	%tmp1 = call i32 @strlen( ptr %tag )
	%tmp3 = call i32 @strlen( ptr %contents )
	%tmp4 = add i32 %tmp1, 2
	%tmp5 = add i32 %tmp4, %tmp3
	%tmp6 = alloca i8, i32 %tmp5
	%tmp9 = call ptr @strcpy( ptr %tmp6, ptr %tag )
	%tmp6.len = call i32 @strlen( ptr %tmp6 )
	%tmp6.indexed = getelementptr i8, ptr %tmp6, i32 %tmp6.len
	call void @llvm.memcpy.p0.p0.i32(ptr align 1 %tmp6.indexed, ptr align 1 @str215, i32 2, i1 false)
	%tmp15 = call ptr @strcat( ptr %tmp6, ptr %contents )
	call fastcc void @comment_add( ptr %vc, ptr %tmp6 )
	ret void
}

declare i32 @strlen(ptr)

declare ptr @strcat(ptr, ptr)

declare fastcc void @comment_add(ptr, ptr)

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind

declare ptr @strcpy(ptr, ptr)
