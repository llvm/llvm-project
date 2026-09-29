; When constant propagating terminator instructions, the basic block iterator
; was not updated to refer to the final position of the new terminator.  This
; can be bad, f.e. because constproping a terminator can lead to the
; destruction of PHI nodes, which invalidates the iterator!
;
; Fixed by adding new arguments to ConstantFoldTerminator
;
; RUN: opt < %s -passes=instsimplify

define void @build_tree(i32 %ml) {
; <label>:0
        br label %bb2

bb2:            ; preds = %bb2, %0
        %reg137 = phi i32 [ %reg140, %bb2 ], [ 12, %0 ]         ; <i32> [#uses=1]
        %reg138 = phi i32 [ %reg139, %bb2 ], [ 0, %0 ]          ; <i32> [#uses=1]
        %reg139 = add i32 %reg138, 1            ; <i32> [#uses=1]
        %reg140 = add i32 %reg137, -1           ; <i32> [#uses=1]
        br i1 false, label %bb2, label %bb3

bb3:            ; preds = %bb2
        ret void
}

