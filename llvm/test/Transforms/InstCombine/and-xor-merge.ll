; RUN: opt < %s -instcombine -S | FileCheck %s

; (x&z) ^ (y&z) -> (x^y)&z
define i32 @test1(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: @test1(
; CHECK-NEXT: %tmp61 = xor i32 %x, %y
; CHECK-NEXT: %tmp7 = and i32 %tmp61, %z
; CHECK-NEXT: ret i32 %tmp7
        %tmp3 = and i32 %z, %x
        %tmp6 = and i32 %z, %y
        %tmp7 = xor i32 %tmp3, %tmp6
        ret i32 %tmp7
}

; (x & y) ^ (x|y) -> x^y
define i32 @test2(i32 %x, i32 %y, i32 %z) {
; CHECK-LABEL: @test2(
; CHECK-NEXT: %tmp7 = xor i32 %y, %x
; CHECK-NEXT: ret i32 %tmp7
        %tmp3 = and i32 %y, %x
        %tmp6 = or i32 %y, %x
        %tmp7 = xor i32 %tmp3, %tmp6
        ret i32 %tmp7
}
