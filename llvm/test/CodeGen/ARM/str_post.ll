; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

define i16 @test1(ptr %X, ptr %A) {
; CHECK-LABEL: test1:
; CHECK: strh {{.*}}[{{.*}}], #-4
        %Y = load i32, ptr %X               ; <i32> [#uses=1]
        %tmp1 = trunc i32 %Y to i16             ; <i16> [#uses=1]
        store i16 %tmp1, ptr %A
        %tmp2 = ptrtoint ptr %A to i16         ; <i16> [#uses=1]
        %tmp3 = sub i16 %tmp2, 4                ; <i16> [#uses=1]
        ret i16 %tmp3
}

define i32 @test2(ptr %X, ptr %A) {
; CHECK-LABEL: test2:
; CHECK: str {{.*}}[{{.*}}],
        %Y = load i32, ptr %X               ; <i32> [#uses=1]
        store i32 %Y, ptr %A
        %tmp1 = ptrtoint ptr %A to i32         ; <i32> [#uses=1]
        %tmp2 = sub i32 %tmp1, 4                ; <i32> [#uses=1]
        ret i32 %tmp2
}
