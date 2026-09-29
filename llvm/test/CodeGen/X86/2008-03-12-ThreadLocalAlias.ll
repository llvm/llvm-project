; RUN: llc < %s -relocation-model=pic | FileCheck %s
; PR2137

; ModuleID = '1.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
	%struct.__res_state = type { i32 }
@__resp = thread_local global ptr @_res		; <ptr> [#uses=1]
@_res = global %struct.__res_state zeroinitializer, section ".bss"		; <ptr> [#uses=1]

@__libc_resp = hidden thread_local alias ptr, ptr @__resp		; <ptr> [#uses=2]

define i32 @foo() {
; CHECK-LABEL: foo:
; CHECK: leal    __libc_resp@TLSLD
entry:
	%retval = alloca i32		; <ptr> [#uses=1]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp = load ptr, ptr @__libc_resp, align 4		; <ptr> [#uses=1]
	%tmp1 = getelementptr %struct.__res_state, ptr %tmp, i32 0, i32 0		; <ptr> [#uses=1]
	store i32 0, ptr %tmp1, align 4
	br label %return
return:		; preds = %entry
	%retval2 = load i32, ptr %retval		; <i32> [#uses=1]
	ret i32 %retval2
}

define i32 @bar() {
; CHECK-LABEL: bar:
; CHECK: leal    __libc_resp@TLSLD
entry:
	%retval = alloca i32		; <ptr> [#uses=1]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp = load ptr, ptr @__libc_resp, align 4		; <ptr> [#uses=1]
	%tmp1 = getelementptr %struct.__res_state, ptr %tmp, i32 0, i32 0		; <ptr> [#uses=1]
	store i32 1, ptr %tmp1, align 4
	br label %return
return:		; preds = %entry
	%retval2 = load i32, ptr %retval		; <i32> [#uses=1]
	ret i32 %retval2
}
