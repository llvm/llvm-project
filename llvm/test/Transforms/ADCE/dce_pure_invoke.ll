; RUN: opt < %s -adce -S | grep null

declare i32 @strlen(i8*) readnone

define i32 @test() personality i32 (...)* @__gxx_personality_v0 {
	; invoke of pure function should not be deleted!
	invoke i32 @strlen( i8* null ) readnone
			to label %Cont unwind label %Other		; <i32>:1 [#uses=0]

Cont:		; preds = %0
	ret i32 0

Other:		; preds = %0
         %exn = landingpad {i8*, i32}
                  cleanup
	ret i32 1
}

declare i32 @__gxx_personality_v0(...)
