; THis testcase caused an assertion failure because a PHI node did not have 
; entries for it's postdominator.  But I think this can only happen when the 
; PHI node is dead, so we just avoid patching up dead PHI nodes.

; RUN: opt < %s -adce

target datalayout = "e-p:32:32"

define void @dead_test8() {
entry:
        br label %loopentry

loopentry:              ; preds = %endif, %entry
        %k.1 = phi i32 [ %k.0, %endif ], [ 0, %entry ]          ; <i32> [#uses=1]
        br i1 false, label %no_exit, label %return

no_exit:                ; preds = %loopentry
        br i1 false, label %then, label %else

then:           ; preds = %no_exit
        br label %endif

else:           ; preds = %no_exit
        %dec = add i32 %k.1, -1         ; <i32> [#uses=1]
        br label %endif

endif:          ; preds = %else, %then
        %k.0 = phi i32 [ %dec, %else ], [ 0, %then ]            ; <i32> [#uses=1]
        store i32 2, i32* null
        br label %loopentry

return:         ; preds = %loopentry
        ret void
}

