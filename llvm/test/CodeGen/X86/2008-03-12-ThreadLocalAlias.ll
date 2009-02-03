; RUN: llvm-as < %s | llc -relocation-model=pic | grep TLSGD | count 2
; PR2137

; ModuleID = '1.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
	%struct.__res_state = type { i32 }
@__resp = thread_local global %struct.__res_state* @_res		; <%struct.__res_state**> [#uses=1]
@_res = global %struct.__res_state zeroinitializer, section ".bss"		; <%struct.__res_state*> [#uses=1]

@__libc_resp = hidden alias %struct.__res_state** @__resp		; <%struct.__res_state**> [#uses=2]

define i32 @foo() {
entry:
	%retval = alloca i32		; <i32*> [#uses=1]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp = load %struct.__res_state** @__libc_resp, align 4		; <%struct.__res_state*> [#uses=1]
	%tmp1 = getelementptr %struct.__res_state* %tmp, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 0, i32* %tmp1, align 4
	br label %return
return:		; preds = %entry
	%retval2 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval2
}

define i32 @bar() {
entry:
	%retval = alloca i32		; <i32*> [#uses=1]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp = load %struct.__res_state** @__libc_resp, align 4		; <%struct.__res_state*> [#uses=1]
	%tmp1 = getelementptr %struct.__res_state* %tmp, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 1, i32* %tmp1, align 4
	br label %return
return:		; preds = %entry
	%retval2 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval2
}
