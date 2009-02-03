; RUN: llvm-as < %s | llc -march=x86 | not grep mov
;
; Test the add and load are folded into the store instruction.

@X = internal global i16 0              ; <i16*> [#uses=2]

define void @foo() {
        %tmp.0 = load i16* @X           ; <i16> [#uses=1]
        %tmp.3 = add i16 %tmp.0, 329            ; <i16> [#uses=1]
        store i16 %tmp.3, i16* @X
        ret void
}

