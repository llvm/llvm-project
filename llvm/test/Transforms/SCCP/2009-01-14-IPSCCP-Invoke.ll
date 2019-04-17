; RUN: opt < %s -ipsccp -S | grep "ret i32 42"
; RUN: opt < %s -ipsccp -S | grep "ret i32 undef"
; PR3325

define i32 @main() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
	%tmp1 = invoke i32 @f()
			to label %UnifiedReturnBlock unwind label %lpad

lpad:
        %val = landingpad { i8*, i32 }
                 cleanup
	unreachable

UnifiedReturnBlock:
	ret i32 %tmp1
}

define internal i32 @f() {
       ret i32 42
}

declare i8* @__cxa_begin_catch(i8*) nounwind

declare void @__cxa_end_catch()

declare i32 @__gxx_personality_v0(...)
