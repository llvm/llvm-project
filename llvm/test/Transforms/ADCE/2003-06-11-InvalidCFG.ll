; RUN: opt < %s -passes=adce -disable-output
; RUN: opt < %s -passes=adce -adce-remove-loops -disable-output

@G = external global i32*               ; <i32**> [#uses=1]

declare void @Fn(i32*)

define i32 @main(i32 %argc.1, i8** %argv.1) {
entry:
        br label %endif.42

endif.42:               ; preds = %shortcirc_done.12, %then.66, %endif.42, %entry
        br i1 false, label %endif.65, label %endif.42

then.66:                ; preds = %shortcirc_done.12
        call void @Fn( i32* %tmp.2846 )
        br label %endif.42

endif.65:               ; preds = %endif.42
        %tmp.2846 = load i32*, i32** @G               ; <i32*> [#uses=1]
        br i1 false, label %shortcirc_next.12, label %shortcirc_done.12

shortcirc_next.12:              ; preds = %endif.65
        br label %shortcirc_done.12

shortcirc_done.12:              ; preds = %shortcirc_next.12, %endif.65
        br i1 false, label %then.66, label %endif.42
}

