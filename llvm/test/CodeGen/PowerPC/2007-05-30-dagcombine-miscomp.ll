; RUN: llc -verify-machineinstrs < %s | FileCheck %s
; PR1473

target triple = "powerpc-unknown-linux-gnu"

; CHECK: foo
; CHECK: rlwinm 3, 3, 23, 30, 30
; CHECK: blr

define zeroext i8 @foo(i16 zeroext  %a)   {
        %tmp2 = lshr i16 %a, 10         ; <i16> [#uses=1]
        %tmp23 = trunc i16 %tmp2 to i8          ; <i8> [#uses=1]
        %tmp4 = shl i8 %tmp23, 1                ; <i8> [#uses=1]
        %tmp5 = and i8 %tmp4, 2         ; <i8> [#uses=1]
        ret i8 %tmp5
}

