; RUN: llvm-as < %s | llc -march=alpha | grep zapnot

;demanded bits mess up this mask in a hard to fix way
;define i64 @foo(i64 %y) {
;        %tmp = and i64 %y,  65535
;        %tmp2 = shr i64 %tmp,  i8 3
;        ret i64 %tmp2
;}

define i64 @foo2(i64 %y) {
        %tmp = lshr i64 %y, 3           ; <i64> [#uses=1]
        %tmp2 = and i64 %tmp, 8191              ; <i64> [#uses=1]
        ret i64 %tmp2
}

