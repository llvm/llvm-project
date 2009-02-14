; RUN: llvm-as < %s | opt -inline | llvm-dis | not grep {tail call void @llvm.memcpy.i32}
; PR3550

define internal void @foo(i32* %p, i32* %q) {
	%pp = bitcast i32* %p to i8*
	%qq = bitcast i32* %q to i8*
	tail call void @llvm.memcpy.i32(i8* %pp, i8* %qq, i32 4, i32 1)
	ret void
}

declare void @llvm.memcpy.i32(i8* nocapture, i8* nocapture, i32, i32) nounwind

define i32 @main() {
	%a = alloca i32		; <i32*> [#uses=3]
	%b = alloca i32		; <i32*> [#uses=2]
	store i32 1, i32* %a, align 4
	store i32 0, i32* %b, align 4
	invoke void @foo(i32* %a, i32* %b)
			to label %invcont unwind label %lpad

invcont:
	%retval = load i32* %a, align 4
	ret i32 %retval

lpad:
	%eh_ptr = call i8* @llvm.eh.exception()
	%eh_select = call i32 (i8*, i8*, ...)* @llvm.eh.selector.i32(i8* %eh_ptr, i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*), i8* null)
	unreachable
}

declare i8* @llvm.eh.exception() nounwind

declare i32 @llvm.eh.selector.i32(i8*, i8*, ...) nounwind

declare i32 @__gxx_personality_v0(...)
