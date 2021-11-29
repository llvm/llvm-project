; RUN: opt < %s -passes=globalopt -disable-output

        %struct._list = type { i32*, %struct._list* }
        %struct._play = type { i32, i32*, %struct._list*, %struct._play* }
@nrow = internal global i32 0           ; <i32*> [#uses=2]

define void @make_play() {
entry:
        br label %cond_true16.i

cond_true16.i:          ; preds = %cond_true16.i, %entry
        %low.0.in.i.0 = phi i32* [ @nrow, %entry ], [ null, %cond_true16.i ]            ; <i32*> [#uses=1]
        %low.0.i = load i32, i32* %low.0.in.i.0              ; <i32> [#uses=0]
        br label %cond_true16.i
}

define void @make_wanted() {
entry:
        unreachable
}

define void @get_good_move() {
entry:
        ret void
}

define void @main() {
entry:
        store i32 8, i32* @nrow
        tail call void @make_play( )
        ret void
}

