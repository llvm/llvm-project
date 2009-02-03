; RUN: llvm-as < %s | llc

define i32 @main() {
bb0:
        %reg109 = malloc i32, i32 100           ; <i32*> [#uses=2]
        br label %bb2

bb2:            ; preds = %bb2, %bb0
        %cann-indvar1 = phi i32 [ 0, %bb0 ], [ %add1-indvar1, %bb2 ]            ; <i32> [#uses=2]
        %reg127 = mul i32 %cann-indvar1, 2              ; <i32> [#uses=1]
        %add1-indvar1 = add i32 %cann-indvar1, 1                ; <i32> [#uses=1]
        store i32 999, i32* %reg109
        %cond1015 = icmp sle i32 1, 99          ; <i1> [#uses=1]
        %reg128 = add i32 %reg127, 2            ; <i32> [#uses=0]
        br i1 %cond1015, label %bb2, label %bb4

bb4:            ; preds = %bb4, %bb2
        %cann-indvar = phi i32 [ %add1-indvar, %bb4 ], [ 0, %bb2 ]              ; <i32> [#uses=1]
        %add1-indvar = add i32 %cann-indvar, 1          ; <i32> [#uses=2]
        store i32 333, i32* %reg109
        %reg131 = add i32 %add1-indvar, 3               ; <i32> [#uses=1]
        %cond1017 = icmp ule i32 %reg131, 99            ; <i1> [#uses=1]
        br i1 %cond1017, label %bb4, label %bb5

bb5:            ; preds = %bb4
        ret i32 0
}

