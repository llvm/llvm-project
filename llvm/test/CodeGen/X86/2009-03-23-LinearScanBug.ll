; RUN: llc < %s -mtriple=i386-apple-darwin -O0

define fastcc void @optimize_bit_field() nounwind {
bb4:
        %a = load i32, ptr null             ; <i32> [#uses=1]
        %s = load i32, ptr getelementptr (i32, ptr null, i32 1)         ; <i32> [#uses=1]
        %z = load i32, ptr getelementptr (i32, ptr null, i32 2)         ; <i32> [#uses=1]
        %r = bitcast i32 0 to i32          ; <i32> [#uses=1]
        %q = trunc i32 %z to i8            ; <i8> [#uses=1]
        %b = icmp eq i8 0, %q              ; <i1> [#uses=1]
        br i1 %b, label %bb73, label %bb72

bb72:      ; preds = %bb4
        %f = tail call fastcc i32 @gen_lowpart(i32 %r, i32 %a) nounwind              ; <i32> [#uses=1]
        br label %bb73

bb73:         ; preds = %bb72, %bb4
        %y = phi i32 [ %f, %bb72 ], [ %s, %bb4 ]          ; <i32> [#uses=1]
        store i32 %y, ptr getelementptr (i32, ptr null, i32 3)
        unreachable
}

declare fastcc i32 @gen_lowpart(i32, i32) nounwind
