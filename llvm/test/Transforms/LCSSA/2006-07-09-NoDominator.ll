; RUN: opt < %s -passes=lcssa

	%struct.SetJmpMapEntry = type { ptr, i32, ptr }

define void @__llvm_sjljeh_try_catching_longjmp_exception() {
entry:
	br label %loopentry
loopentry:		; preds = %endif, %entry
	%SJE.0 = phi ptr [ null, %entry ], [ %tmp.25, %endif ]	; <ptr> [#uses=1]
	br i1 false, label %no_exit, label %loopexit
no_exit:		; preds = %loopentry
	br i1 false, label %then, label %endif
then:		; preds = %no_exit
	%tmp.21 = getelementptr %struct.SetJmpMapEntry, ptr %SJE.0, i32 0, i32 1		; <ptr> [#uses=0]
	br label %return
endif:		; preds = %no_exit
	%tmp.25 = load ptr, ptr null		; <ptr> [#uses=1]
	br label %loopentry
loopexit:		; preds = %loopentry
	br label %return
return:		; preds = %loopexit, %then
	ret void
}

