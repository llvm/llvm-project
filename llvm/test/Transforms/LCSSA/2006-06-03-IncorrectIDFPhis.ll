; RUN: opt < %s -passes=loop-simplify,lcssa -S | FileCheck %s

        %struct.SetJmpMapEntry = type { ptr, i32, ptr }

define void @__llvm_sjljeh_try_catching_longjmp_exception() {
; CHECK-LABEL: @__llvm_sjljeh_try_catching_longjmp_exception
entry:
        br i1 false, label %UnifiedReturnBlock, label %no_exit
no_exit:                ; preds = %endif, %entry
        %SJE.0.0 = phi ptr [ %tmp.24, %endif ], [ null, %entry ]            ; <ptr> [#uses=1]
        br i1 false, label %then, label %endif
then:           ; preds = %no_exit
; CHECK: %SJE.0.0.lcssa = phi ptr
        %tmp.20 = getelementptr %struct.SetJmpMapEntry, ptr %SJE.0.0, i32 0, i32 1          ; <ptr> [#uses=0]
        ret void
endif:          ; preds = %no_exit
        %tmp.24 = load ptr, ptr null            ; <ptr> [#uses=1]
        br i1 false, label %UnifiedReturnBlock, label %no_exit
UnifiedReturnBlock:             ; preds = %endif, %entry
        ret void
}

