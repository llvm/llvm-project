; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep and | count 1
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep xor | count 2

; (x&z) ^ (y&z) -> (x^y)&z
define i32 @test1(i32 %x, i32 %y, i32 %z) {
        %tmp3 = and i32 %z, %x
        %tmp6 = and i32 %z, %y
        %tmp7 = xor i32 %tmp3, %tmp6
        ret i32 %tmp7
}

; (x & y) ^ (x|y) -> x^y
define i32 @test2(i32 %x, i32 %y, i32 %z) {
        %tmp3 = and i32 %y, %x
        %tmp6 = or i32 %y, %x
        %tmp7 = xor i32 %tmp3, %tmp6
        ret i32 %tmp7
}

