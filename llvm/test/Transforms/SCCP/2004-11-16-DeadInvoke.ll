; RUN: opt < %s -passes=sccp -disable-output

declare i32 @foo()

define void @caller() personality ptr @__gxx_personality_v0 {
	br i1 true, label %T, label %F
F:		; preds = %0
	%X = invoke i32 @foo( )
			to label %T unwind label %LP		; <i32> [#uses=0]
LP:
        %val = landingpad { ptr, i32 }
                 catch ptr null
        br label %T
T:
	ret void
}

declare i32 @__gxx_personality_v0(...)
