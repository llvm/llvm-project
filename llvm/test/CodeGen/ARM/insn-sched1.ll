; RUN: llc -mtriple=arm-eabi -mattr=+v6 %s -o /dev/null
; RUN: llc -mtriple=arm-apple-ios -mattr=+v6 %s -o - | FileCheck %s

define i32 @test(i32 %x) "no-frame-pointer-elim"="true" {
        %tmp = trunc i32 %x to i16              ; <i16> [#uses=1]
        %tmp2 = call i32 @f( i32 1, i16 %tmp )             ; <i32> [#uses=1]
        ret i32 %tmp2
}

declare i32 @f(i32, i16)

; CHECK: mov
; CHECK: mov
; CHECK: mov
; CHECK-NOT: mov

