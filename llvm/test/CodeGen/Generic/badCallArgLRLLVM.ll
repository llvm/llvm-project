; RUN: llc < %s

; This caused a problem because the argument of a call was defined by
; the return value of another call that appears later in the code.
; When processing the first call, the second call has not yet been processed
; so no LiveRange has been created for its return value.
; 
; llc dies in UltraSparcRegInfo::suggestRegs4CallArgs() with:
;     ERROR: In call instr, no LR for arg: 0x1009e0740 
;

declare i32 @getInt(i32)

define i32 @main(i32 %argc, ptr %argv) {
bb0:
        br label %bb2

bb1:            ; preds = %bb2
        %reg222 = call i32 @getInt( i32 %reg218 )               ; <i32> [#uses=1]
        %reg110 = add i32 %reg222, 1            ; <i32> [#uses=2]
        %b = icmp sle i32 %reg110, 0            ; <i1> [#uses=1]
        br i1 %b, label %bb2, label %bb3

bb2:            ; preds = %bb1, %bb0
        %reg218 = call i32 @getInt( i32 %argc )         ; <i32> [#uses=1]
        br label %bb1

bb3:            ; preds = %bb1
        ret i32 %reg110
}

