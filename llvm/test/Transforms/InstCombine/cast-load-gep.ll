; RUN: llvm-as < %s | opt -instcombine -globaldce | llvm-dis | \
; RUN:   not grep Array

; Pulling the cast out of the load allows us to eliminate the load, and then 
; the whole array.

        %op = type { float }
        %unop = type { i32 }
@Array = internal constant [1 x %op* (%op*)*] [ %op* (%op*)* @foo ]             ; <[1 x %op* (%op*)*]*> [#uses=1]

define %op* @foo(%op* %X) {
        ret %op* %X
}

define %unop* @caller(%op* %O) {
        %tmp = load %unop* (%op*)** bitcast ([1 x %op* (%op*)*]* @Array to %unop* (%op*)**); <%unop* (%op*)*> [#uses=1]
        %tmp.2 = call %unop* %tmp( %op* %O )            ; <%unop*> [#uses=1]
        ret %unop* %tmp.2
}

