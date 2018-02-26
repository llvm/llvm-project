; RUN: llc < %s -mtriple=i686-- -x86-asm-syntax=intel | \
; RUN:   grep "shld.*cl"
; RUN: llc < %s -mtriple=i686-- -x86-asm-syntax=intel | \
; RUN:   not grep "mov cl, bl"

; PR687

define i64 @foo(i64 %x, i64* %X) {
        %tmp.1 = load i64, i64* %X           ; <i64> [#uses=1]
        %tmp.3 = trunc i64 %tmp.1 to i8         ; <i8> [#uses=1]
        %shift.upgrd.1 = zext i8 %tmp.3 to i64          ; <i64> [#uses=1]
        %tmp.4 = shl i64 %x, %shift.upgrd.1             ; <i64> [#uses=1]
        ret i64 %tmp.4
}

