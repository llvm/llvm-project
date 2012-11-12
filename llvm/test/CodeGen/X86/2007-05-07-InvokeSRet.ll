; RUN: llc < %s -mtriple=i686-pc-linux-gnu -disable-fp-elim | not grep "addl .12, %esp"
; PR1398

	%struct.S = type { i32, i32 }

declare void @invokee(%struct.S* sret )

define void @invoker(%struct.S* %name.0.0) {
entry:
	invoke void @invokee( %struct.S* sret %name.0.0   )
			to label %return unwind label %return

return:		; preds = %entry, %entry
        %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
                 cleanup
	ret void
}

declare i32 @__gxx_personality_v0(...)
