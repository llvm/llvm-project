; RUN: llvm-as < %s | llc -march=c

; This case was emitting code that looked like this:
; ...
;   llvm_BB1:       /* no statement here */
; }
; 
; Which the Sun C compiler rejected, so now we are sure to put a return 
; instruction in there if the basic block is otherwise empty.
;
define void @test() {
        br label %BB1

BB2:            ; preds = %BB2
        br label %BB2

BB1:            ; preds = %0
        ret void
}

