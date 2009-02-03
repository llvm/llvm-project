; RUN: llvm-as < %s | llc -march=arm > %t
; RUN: grep __divsi3  %t
; RUN: grep __udivsi3 %t
; RUN: grep __modsi3  %t
; RUN: grep __umodsi3 %t

define i32 @f1(i32 %a, i32 %b) {
entry:
        %tmp1 = sdiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f2(i32 %a, i32 %b) {
entry:
        %tmp1 = udiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f3(i32 %a, i32 %b) {
entry:
        %tmp1 = srem i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f4(i32 %a, i32 %b) {
entry:
        %tmp1 = urem i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

