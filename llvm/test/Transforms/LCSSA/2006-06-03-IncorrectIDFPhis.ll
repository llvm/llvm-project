; RUN: opt < %s -loop-simplify -lcssa -S | FileCheck %s

        %struct.SetJmpMapEntry = type { i8*, i32, %struct.SetJmpMapEntry* }

define void @__llvm_sjljeh_try_catching_longjmp_exception() {
; CHECK-LABEL: @__llvm_sjljeh_try_catching_longjmp_exception
entry:
        br i1 false, label %UnifiedReturnBlock, label %no_exit
no_exit:                ; preds = %endif, %entry
        %SJE.0.0 = phi %struct.SetJmpMapEntry* [ %tmp.24, %endif ], [ null, %entry ]            ; <%struct.SetJmpMapEntry*> [#uses=1]
        br i1 false, label %then, label %endif
then:           ; preds = %no_exit
; CHECK: %SJE.0.0.lcssa = phi %struct.SetJmpMapEntry
        %tmp.20 = getelementptr %struct.SetJmpMapEntry, %struct.SetJmpMapEntry* %SJE.0.0, i32 0, i32 1          ; <i32*> [#uses=0]
        ret void
endif:          ; preds = %no_exit
        %tmp.24 = load %struct.SetJmpMapEntry*, %struct.SetJmpMapEntry** null            ; <%struct.SetJmpMapEntry*> [#uses=1]
        br i1 false, label %UnifiedReturnBlock, label %no_exit
UnifiedReturnBlock:             ; preds = %endif, %entry
        ret void
}

