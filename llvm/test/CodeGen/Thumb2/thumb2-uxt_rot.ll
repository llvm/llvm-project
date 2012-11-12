; RUN: llc < %s -march=thumb -mcpu=cortex-a8 | FileCheck %s --check-prefix=A8
; RUN: llc < %s -march=thumb -mcpu=cortex-m3 | FileCheck %s --check-prefix=M3
; rdar://11318438

define zeroext i8 @test1(i32 %A.u)  {
; A8: test1
; A8: uxtb r0, r0
    %B.u = trunc i32 %A.u to i8
    ret i8 %B.u
}

define zeroext i32 @test2(i32 %A.u, i32 %B.u)  {
; A8: test2
; A8: uxtab  r0, r0, r1

; M3: test2
; M3: uxtb  r1, r1
; M3-NOT: uxtab
; M3: add   r0, r1
    %C.u = trunc i32 %B.u to i8
    %D.u = zext i8 %C.u to i32
    %E.u = add i32 %A.u, %D.u
    ret i32 %E.u
}

define zeroext i32 @test3(i32 %A.u)  {
; A8: test3
; A8: uxth.w r0, r0, ror #8
    %B.u = lshr i32 %A.u, 8
    %C.u = shl i32 %A.u, 24
    %D.u = or i32 %B.u, %C.u
    %E.u = trunc i32 %D.u to i16
    %F.u = zext i16 %E.u to i32
    ret i32 %F.u
}
