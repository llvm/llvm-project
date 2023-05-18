; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

define i32 @simpleAdd(i32 %i0, i32 %j0) {
        %t1 = xor i32 %i0, %j0          ; <i32> [#uses=1]
        %t2 = or i32 %i0, %j0           ; <i32> [#uses=1]
        %t3 = and i32 %t1, %t2          ; <i32> [#uses=1]
        ret i32 %t3
}

