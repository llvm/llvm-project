; Do not remove the invoke!
;
; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -disable-output

define i32 @test() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
	%A = invoke i32 @test( )
			to label %Ret unwind label %Ret2		; <i32> [#uses=1]
Ret:		; preds = %0
	ret i32 %A
Ret2:		; preds = %0
        %val = landingpad { i8*, i32 }
                 catch i8* null
	ret i32 undef
}

declare i32 @__gxx_personality_v0(...)
