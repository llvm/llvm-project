; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s

; %inc2 uses it's own value, but that's ok, as it's unreachable!

define void @test() {
entry:
        ret void

no_exit.2:              ; preds = %endif.6
        %tmp.103 = fcmp olt double 0.000000e+00, 0.000000e+00           ; <i1> [#uses=1]
        br i1 %tmp.103, label %endif.6, label %else.0

else.0:         ; preds = %no_exit.2
        store i16 0, ptr null
        br label %endif.6

endif.6:                ; preds = %else.0, %no_exit.2
        %inc.2 = add i32 %inc.2, 1              ; <i32> [#uses=2]
        %tmp.96 = icmp slt i32 %inc.2, 0                ; <i1> [#uses=1]
        br i1 %tmp.96, label %no_exit.2, label %UnifiedReturnBlock1

UnifiedReturnBlock1:            ; preds = %endif.6
        ret void
}

