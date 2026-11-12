; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

define i16 @FunFunc(i64 %x, i8 %z) {
bb0:
        %cast110 = sext i8 %z to i16            ; <i16> [#uses=1]
        %cast10 = trunc i64 %x to i16           ; <i16> [#uses=1]
        %reg109 = add i16 %cast110, %cast10             ; <i16> [#uses=1]
        ret i16 %reg109
}

