; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define i32 @test(i32 %a, i32 %b, i32 %c) {
        %tmp1 = mul i32 %a, %b          ; <i32> [#uses=2]
        %tmp2 = inttoptr i32 %tmp1 to ptr              ; <ptr> [#uses=1]
        %tmp3 = load i32, ptr %tmp2         ; <i32> [#uses=1]
        %tmp4 = sub i32 %tmp1, 8               ; <i32> [#uses=1]
        %tmp5 = mul i32 %tmp4, %tmp3            ; <i32> [#uses=1]
        ret i32 %tmp5
}
; CHECK: 	ldr	r{{.*}},	[{{.*}}],

