; RUN: llvm-as < %s | llc -march=x86 | grep add
; RUN: llvm-as < %s | llc -march=x86 | not grep adc

define i64 @test(i64 %A, i32 %B) {
        %tmp12 = zext i32 %B to i64             ; <i64> [#uses=1]
        %tmp3 = shl i64 %tmp12, 32              ; <i64> [#uses=1]
        %tmp5 = add i64 %tmp3, %A               ; <i64> [#uses=1]
        ret i64 %tmp5
}

