; RUN: opt < %s -simplifycfg -disable-output

define i1 @foo() {
	%X = invoke i1 @foo( )
			to label %N unwind label %F		; <i1> [#uses=1]
F:		; preds = %0
        %val = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
                 catch i8* null
	ret i1 false
N:		; preds = %0
	br i1 %X, label %A, label %B
A:		; preds = %N
	ret i1 true
B:		; preds = %N
	ret i1 true
}

declare i32 @__gxx_personality_v0(...)
