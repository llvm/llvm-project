; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2,+dsp %s -o - | FileCheck %s
; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2,+dsp -arm-use-mulops=false %s -o - | FileCheck %s -check-prefix=NO_MULOPS

define i32 @f3(i32 %a, i16 %x, i32 %y) {
; CHECK: f3
; CHECK: smlabt r0, r1, r2, r0
; NO_MULOPS: f3
; NO_MULOPS: smultb r1, r2, r1
; NO_MULOPS-NEXT: add r0, r1
        %tmp = sext i16 %x to i32               ; <i32> [#uses=1]
        %tmp2 = ashr i32 %y, 16         ; <i32> [#uses=1]
        %tmp3 = mul i32 %tmp2, %tmp             ; <i32> [#uses=1]
        %tmp5 = add i32 %tmp3, %a               ; <i32> [#uses=1]
        ret i32 %tmp5
}
