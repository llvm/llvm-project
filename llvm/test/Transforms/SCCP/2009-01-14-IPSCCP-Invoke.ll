; RUN: llvm-as < %s | opt -ipsccp | llvm-dis | grep {ret i32 42}
; RUN: llvm-as < %s | opt -ipsccp | llvm-dis | grep {ret i32 undef}
; PR3325

define i32 @main() {
	%tmp1 = invoke i32 @f()
			to label %UnifiedReturnBlock unwind label %lpad

lpad:
	unreachable

UnifiedReturnBlock:
	ret i32 %tmp1
}

define internal i32 @f() {
       ret i32 42
}

declare i8* @__cxa_begin_catch(i8*) nounwind

declare i8* @llvm.eh.exception() nounwind

declare i32 @llvm.eh.selector.i32(i8*, i8*, ...) nounwind

declare void @__cxa_end_catch()

declare i32 @__gxx_personality_v0(...)
