; RUN: llc < %s -march=arm | FileCheck %s
; RUN: llc < %s -march=arm -mcpu=swift | FileCheck %s

; CHECK: test1:
; CHECK: ldr {{.*, \[.*]}}, -r2
; CHECK-NOT: ldr
define i32 @test1(i32 %a, i32 %b, i32 %c) {
        %tmp1 = mul i32 %a, %b          ; <i32> [#uses=2]
        %tmp2 = inttoptr i32 %tmp1 to i32*              ; <i32*> [#uses=1]
        %tmp3 = load i32* %tmp2         ; <i32> [#uses=1]
        %tmp4 = sub i32 %tmp1, %c               ; <i32> [#uses=1]
        %tmp5 = mul i32 %tmp4, %tmp3            ; <i32> [#uses=1]
        ret i32 %tmp5
}

; CHECK: test2:
; CHECK: ldr {{.*, \[.*\]}}, #-16
; CHECK-NOT: ldr
define i32 @test2(i32 %a, i32 %b) {
        %tmp1 = mul i32 %a, %b          ; <i32> [#uses=2]
        %tmp2 = inttoptr i32 %tmp1 to i32*              ; <i32*> [#uses=1]
        %tmp3 = load i32* %tmp2         ; <i32> [#uses=1]
        %tmp4 = sub i32 %tmp1, 16               ; <i32> [#uses=1]
        %tmp5 = mul i32 %tmp4, %tmp3            ; <i32> [#uses=1]
        ret i32 %tmp5
}
