; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -o %t -f
; RUN: grep pshuflw %t | count 1
; RUN: grep pextrw %t | count 2
; RUN: grep pinsrw %t | count 2
; PR2585

; FIXME: This testcase produces icky code. It can be made much better!

external constant <4 x i32>             ; <<4 x i32>*>:0 [#uses=1]
external constant <4 x i16>             ; <<4 x i16>*>:1 [#uses=1]

define internal void @""() {
        load <4 x i32>* @0, align 16            ; <<4 x i32>>:1 [#uses=1]
        bitcast <4 x i32> %1 to <8 x i16>               ; <<8 x i16>>:2[#uses=1]
        shufflevector <8 x i16> %2, <8 x i16> undef, <8 x i32> < i32 0, i32 2, i32 4, i32 6, i32 undef, i32 undef, i32 undef, i32 undef >               ; <<8x i16>>:3 [#uses=1]
        bitcast <8 x i16> %3 to <2 x i64>               ; <<2 x i64>>:4 [#uses=1]
        extractelement <2 x i64> %4, i32 0              ; <i64>:5 [#uses=1]
        bitcast i64 %5 to <4 x i16>             ; <<4 x i16>>:6 [#uses=1]
        store <4 x i16> %6, <4 x i16>* @1, align 8
        ret void
}
