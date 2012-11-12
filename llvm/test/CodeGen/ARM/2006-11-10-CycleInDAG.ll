; RUN: llc < %s -march=arm -mattr=+v6

%struct.layer_data = type { i32, [2048 x i8], i8*, [16 x i8], i32, i8*, i32, i32, [64 x i32], [64 x i32], [64 x i32], [64 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [12 x [64 x i16]] }
@ld = external global %struct.layer_data*               ; <%struct.layer_data**> [#uses=1]

define void @main() {
entry:
        br i1 false, label %bb169.i, label %cond_true11

bb169.i:                ; preds = %entry
        ret void

cond_true11:            ; preds = %entry
        %tmp.i32 = load %struct.layer_data** @ld                ; <%struct.layer_data*> [#uses=2]
        %tmp3.i35 = getelementptr %struct.layer_data* %tmp.i32, i32 0, i32 1, i32 2048; <i8*> [#uses=2]
        %tmp.i36 = getelementptr %struct.layer_data* %tmp.i32, i32 0, i32 2          ; <i8**> [#uses=1]
        store i8* %tmp3.i35, i8** %tmp.i36
        store i8* %tmp3.i35, i8** null
        ret void
}
