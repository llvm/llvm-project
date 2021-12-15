; The PHI cannot be eliminated from this testcase, SCCP is mishandling invoke's!
; RUN: opt < %s -passes=sccp -S | grep phi

declare void @foo()

define i32 @test(i1 %cond) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
Entry:
	br i1 %cond, label %Inv, label %Cont
Inv:		; preds = %Entry
	invoke void @foo( )
			to label %Ok unwind label %LPad
Ok:		; preds = %Inv
	br label %Cont
LPad:
        %val = landingpad { i8*, i32 }
                 catch i8* null
        br label %Cont
Cont:		; preds = %Ok, %Inv, %Entry
	%X = phi i32 [ 0, %Entry ], [ 1, %Ok ], [ 0, %LPad ]		; <i32> [#uses=1]
	ret i32 %X
}

declare i32 @__gxx_personality_v0(...)
