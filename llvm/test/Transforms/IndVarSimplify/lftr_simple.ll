; LFTR should eliminate the need for the computation of i*i completely.  It 
; is only used to compute the exit value.
; RUN: opt < %s -indvars -dce -S | not grep mul

@A = external global i32                ; <i32*> [#uses=1]

define i32 @quadratic_setlt() {
entry:
        br label %loop

loop:           ; preds = %loop, %entry
        %i = phi i32 [ 7, %entry ], [ %i.next, %loop ]          ; <i32> [#uses=5]
        %i.next = add i32 %i, 1         ; <i32> [#uses=1]
        store i32 %i, i32* @A
        %i2 = mul i32 %i, %i            ; <i32> [#uses=1]
        %c = icmp slt i32 %i2, 1000             ; <i1> [#uses=1]
        br i1 %c, label %loop, label %loopexit

loopexit:               ; preds = %loop
        ret i32 %i
}

