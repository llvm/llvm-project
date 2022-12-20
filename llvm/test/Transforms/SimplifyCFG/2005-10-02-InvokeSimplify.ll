; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -disable-output

define i1 @foo() personality ptr @__gxx_personality_v0 {
	%X = invoke i1 @foo( )
			to label %N unwind label %F		; <i1> [#uses=1]
F:		; preds = %0
        %val = landingpad { ptr, i32 }
                 catch ptr null
	ret i1 false
N:		; preds = %0
	br i1 %X, label %A, label %B
A:		; preds = %N
	ret i1 true
B:		; preds = %N
	ret i1 true
}

declare i32 @__gxx_personality_v0(...)
