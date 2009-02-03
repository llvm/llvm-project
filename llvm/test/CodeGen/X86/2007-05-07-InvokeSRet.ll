; RUN: llvm-as < %s | llc -mtriple=i686-pc-linux-gnu -enable-eh -disable-fp-elim | not grep {addl .12, %esp}
; PR1398

	%struct.S = type { i32, i32 }

declare void @invokee(%struct.S* sret )

define void @invoker(%struct.S* %name.0.0) {
entry:
	invoke void @invokee( %struct.S* %name.0.0 sret  )
			to label %return unwind label %return

return:		; preds = %entry, %entry
	ret void
}
