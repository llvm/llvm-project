; RUN: llc -mtriple=arm-eabi -mattr=+v6 %s -o /dev/null

%struct.layer_data = type { i32, [2048 x i8], ptr, [16 x i8], i32, ptr, i32, i32, [64 x i32], [64 x i32], [64 x i32], [64 x i32], i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [12 x [64 x i16]] }
@ld = external global ptr               ; <ptr> [#uses=1]

define void @main() {
entry:
        br i1 false, label %bb169.i, label %cond_true11

bb169.i:                ; preds = %entry
        ret void

cond_true11:            ; preds = %entry
        %tmp.i32 = load ptr, ptr @ld                ; <ptr> [#uses=2]
        %tmp3.i35 = getelementptr %struct.layer_data, ptr %tmp.i32, i32 0, i32 1, i32 2048; <ptr> [#uses=2]
        %tmp.i36 = getelementptr %struct.layer_data, ptr %tmp.i32, i32 0, i32 2          ; <ptr> [#uses=1]
        store ptr %tmp3.i35, ptr %tmp.i36
        store ptr %tmp3.i35, ptr null
        ret void
}
