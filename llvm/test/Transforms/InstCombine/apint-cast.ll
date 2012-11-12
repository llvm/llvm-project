; Tests to make sure elimination of casts is working correctly
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define i17 @test1(i17 %a) {
        %tmp = zext i17 %a to i37               ; <i37> [#uses=2]
        %tmp21 = lshr i37 %tmp, 8               ; <i37> [#uses=1]
; CHECK: %tmp21 = lshr i17 %a, 8
        %tmp5 = shl i37 %tmp, 8         ; <i37> [#uses=1]
; CHECK: %tmp5 = shl i17 %a, 8
        %tmp.upgrd.32 = or i37 %tmp21, %tmp5            ; <i37> [#uses=1]
; CHECK: %tmp.upgrd.32 = or i17 %tmp21, %tmp5
        %tmp.upgrd.3 = trunc i37 %tmp.upgrd.32 to i17   ; <i17> [#uses=1]
        ret i17 %tmp.upgrd.3
; CHECK: ret i17 %tmp.upgrd.32
}

define i167 @test2(i167 %a) {
        %tmp = zext i167 %a to i577               ; <i577> [#uses=2]
        %tmp21 = lshr i577 %tmp, 9               ; <i577> [#uses=1]
; CHECK: %tmp21 = lshr i167 %a, 9
        %tmp5 = shl i577 %tmp, 8         ; <i577> [#uses=1]
; CHECK: %tmp5 = shl i167 %a, 8
        %tmp.upgrd.32 = or i577 %tmp21, %tmp5            ; <i577> [#uses=1]
; CHECK: %tmp.upgrd.32 = or i167 %tmp21, %tmp5
        %tmp.upgrd.3 = trunc i577 %tmp.upgrd.32 to i167  ; <i167> [#uses=1]
        ret i167 %tmp.upgrd.3
; CHECK: ret i167 %tmp.upgrd.32
}
