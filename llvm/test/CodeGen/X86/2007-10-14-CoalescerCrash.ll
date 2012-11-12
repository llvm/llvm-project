; RUN: llc < %s -mtriple=x86_64-apple-darwin

        %struct._Unwind_Context = type {  }

define i32 @execute_stack_op(i8* %op_ptr, i8* %op_end, %struct._Unwind_Context* %context, i64 %initial) {
entry:
        br i1 false, label %bb, label %return

bb:             ; preds = %bb31, %entry
        br i1 false, label %bb6, label %bb31

bb6:            ; preds = %bb
        %tmp10 = load i64* null, align 8                ; <i64> [#uses=1]
        %tmp16 = load i64* null, align 8                ; <i64> [#uses=1]
        br i1 false, label %bb23, label %bb31

bb23:           ; preds = %bb6
        %tmp2526.cast = and i64 %tmp16, 4294967295              ; <i64> [#uses=1]
        %tmp27 = ashr i64 %tmp10, %tmp2526.cast         ; <i64> [#uses=1]
        br label %bb31

bb31:           ; preds = %bb23, %bb6, %bb
        %result.0 = phi i64 [ %tmp27, %bb23 ], [ 0, %bb ], [ 0, %bb6 ]          ; <i64> [#uses=0]
        br i1 false, label %bb, label %return

return:         ; preds = %bb31, %entry
        ret i32 undef
}
