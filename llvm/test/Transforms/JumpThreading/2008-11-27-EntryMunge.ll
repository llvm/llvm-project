; RUN: opt < %s -jump-threading -S | grep "ret i32 0"
; PR3138

define i32 @jt() {
entry:
       br i1 true, label %bb3, label %bb

bb:             ; preds = %entry
       unreachable

bb3:            ; preds = %entry
       ret i32 0
}
