; RUN: llc < %s -march=mips
; PR2794

define i32 @main(ptr) nounwind {
entry:
        br label %continue.outer

continue.outer:         ; preds = %case4, %entry
        %p.0.ph.rec = phi i32 [ 0, %entry ], [ %indvar.next, %case4 ]          ; <i32> [#uses=2]
        %p.0.ph = getelementptr i8, ptr %0, i32 %p.0.ph.rec         ; <ptr> [#uses=1]
        %1 = load i8, ptr %p.0.ph           ; <i8> [#uses=1]
        switch i8 %1, label %infloop [
                i8 0, label %return.split
                i8 76, label %case4
                i8 108, label %case4
                i8 104, label %case4
                i8 42, label %case4
        ]

case4:          ; preds = %continue.outer, %continue.outer, %continue.outer, %continue.outer
        %indvar.next = add i32 %p.0.ph.rec, 1           ; <i32> [#uses=1]
        br label %continue.outer

return.split:           ; preds = %continue.outer
        ret i32 0

infloop:                ; preds = %infloop, %continue.outer
        br label %infloop
}
