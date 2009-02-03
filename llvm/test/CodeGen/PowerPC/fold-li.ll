; RUN: llvm-as < %s | llc -march=ppc32  | \
; RUN:   grep -v align | not grep li

;; Test that immediates are folded into these instructions correctly.

define i32 @ADD(i32 %X) nounwind {
        %Y = add i32 %X, 65537          ; <i32> [#uses=1]
        ret i32 %Y
}

define i32 @SUB(i32 %X) nounwind {
        %Y = sub i32 %X, 65537          ; <i32> [#uses=1]
        ret i32 %Y
}

