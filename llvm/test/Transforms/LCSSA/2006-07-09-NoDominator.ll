; RUN: opt < %s -lcssa

	%struct.SetJmpMapEntry = type { i8*, i32, %struct.SetJmpMapEntry* }

define void @__llvm_sjljeh_try_catching_longjmp_exception() {
entry:
	br label %loopentry
loopentry:		; preds = %endif, %entry
	%SJE.0 = phi %struct.SetJmpMapEntry* [ null, %entry ], [ %tmp.25, %endif ]	; <%struct.SetJmpMapEntry*> [#uses=1]
	br i1 false, label %no_exit, label %loopexit
no_exit:		; preds = %loopentry
	br i1 false, label %then, label %endif
then:		; preds = %no_exit
	%tmp.21 = getelementptr %struct.SetJmpMapEntry* %SJE.0, i32 0, i32 1		; <i32*> [#uses=0]
	br label %return
endif:		; preds = %no_exit
	%tmp.25 = load %struct.SetJmpMapEntry** null		; <%struct.SetJmpMapEntry*> [#uses=1]
	br label %loopentry
loopexit:		; preds = %loopentry
	br label %return
return:		; preds = %loopexit, %then
	ret void
}

