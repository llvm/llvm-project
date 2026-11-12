; RUN: llc -verify-machineinstrs < %s -mcpu=g5 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mattr=+altivec -mattr=-vsx -mattr=-power8-vector | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu -mattr=+altivec -mattr=-vsx -mattr=-power8-vector | FileCheck %s -check-prefix=CHECK-LE

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc-unknown-linux-gnu"
	%struct.S2203 = type { %struct.u16qi }
	%struct.u16qi = type { <16 x i8> }
@s = weak global %struct.S2203 zeroinitializer		; <ptr> [#uses=1]

define void @foo(i32 %x, ...) {
entry:
; CHECK: foo:
; CHECK-LE: foo:
	%x_addr = alloca i32		; <ptr> [#uses=1]
	%ap = alloca ptr		; <ptr> [#uses=3]
	%ap.0 = alloca ptr		; <ptr> [#uses=3]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %x, ptr %x_addr
	call void @llvm.va_start( ptr %ap )
	%tmp = load ptr, ptr %ap, align 4		; <ptr> [#uses=1]
	store ptr %tmp, ptr %ap.0, align 4
	%tmp2 = load ptr, ptr %ap.0, align 4		; <ptr> [#uses=1]
	%tmp3 = getelementptr i8, ptr %tmp2, i64 16		; <ptr> [#uses=1]
	store ptr %tmp3, ptr %ap, align 4
	%tmp4 = load ptr, ptr %ap.0, align 4		; <ptr> [#uses=1]
	%tmp6 = getelementptr %struct.S2203, ptr @s, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp7 = getelementptr %struct.S2203, ptr %tmp4, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp8 = getelementptr %struct.u16qi, ptr %tmp6, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp9 = getelementptr %struct.u16qi, ptr %tmp7, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp10 = load <16 x i8>, ptr %tmp9, align 4		; <<16 x i8>> [#uses=1]
; CHECK: lvsl
; CHECK: vperm
; CHECK-LE: lvsr
; CHECK-LE: vperm
	store <16 x i8> %tmp10, ptr %tmp8, align 4
	br label %return

return:		; preds = %entry
	ret void
}

declare void @llvm.va_start(ptr) nounwind 
