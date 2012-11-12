; RUN: llc < %s -march=arm | FileCheck %s
; RUN: llc < %s -march=arm -mcpu=swift | FileCheck %s

; CHECK: test1:
; CHECK: ldr {{.*!}}
; CHECK-NOT: ldr
define i32* @test1(i32* %X, i32* %dest) {
        %Y = getelementptr i32* %X, i32 4               ; <i32*> [#uses=2]
        %A = load i32* %Y               ; <i32> [#uses=1]
        store i32 %A, i32* %dest
        ret i32* %Y
}

; CHECK: test2:
; CHECK: ldr {{.*!}}
; CHECK-NOT: ldr
define i32 @test2(i32 %a, i32 %b, i32 %c) {
        %tmp1 = sub i32 %a, %b          ; <i32> [#uses=2]
        %tmp2 = inttoptr i32 %tmp1 to i32*              ; <i32*> [#uses=1]
        %tmp3 = load i32* %tmp2         ; <i32> [#uses=1]
        %tmp4 = sub i32 %tmp1, %c               ; <i32> [#uses=1]
        %tmp5 = add i32 %tmp4, %tmp3            ; <i32> [#uses=1]
        ret i32 %tmp5
}
