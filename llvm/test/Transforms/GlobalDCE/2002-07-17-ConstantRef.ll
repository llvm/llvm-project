; RUN: llvm-as < %s | opt -globaldce
;

@X = global void ()* @func              ; <void ()**> [#uses=0]

; Not dead, can be reachable via X
define internal void @func() {
        ret void
}

define void @main() {
        ret void
}
