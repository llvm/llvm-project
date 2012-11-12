; RUN: llc < %s -march=ppc32 -mtriple=powerpc-apple-darwin8 | \
; RUN:   grep "srwi r., r., 5"

define i32 @eq0(i32 %a) {
        %tmp.1 = icmp eq i32 %a, 0              ; <i1> [#uses=1]
        %tmp.2 = zext i1 %tmp.1 to i32          ; <i32> [#uses=1]
        ret i32 %tmp.2
}

