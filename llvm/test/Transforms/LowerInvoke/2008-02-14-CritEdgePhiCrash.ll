; RUN: llvm-as < %s | opt -lowerinvoke -enable-correct-eh-support -disable-output
; PR2029
define i32 @main(i32 %argc, i8** %argv) {
bb470:
        invoke i32 @main(i32 0, i8** null) to label %invcont474 unwind label
%lpad902

invcont474:             ; preds = %bb470
        ret i32 0

lpad902:                ; preds = %bb470
        %tmp471.lcssa = phi i8* [ null, %bb470 ]                ; <i8*>
        ret i32 0
}
