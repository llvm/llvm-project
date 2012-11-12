; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128:n8:16:32:64"

define i32 @test1(i32 %x) {
        %tmp.1 = and i32 %x, 65535              ; <i32> [#uses=1]
        %tmp.2 = xor i32 %tmp.1, -32768         ; <i32> [#uses=1]
        %tmp.3 = add i32 %tmp.2, 32768          ; <i32> [#uses=1]
        ret i32 %tmp.3
; CHECK: @test1
; CHECK: %sext = shl i32 %x, 16
; CHECK: %tmp.3 = ashr exact i32 %sext, 16
; CHECK: ret i32 %tmp.3
}

define i32 @test2(i32 %x) {
        %tmp.1 = and i32 %x, 65535              ; <i32> [#uses=1]
        %tmp.2 = xor i32 %tmp.1, 32768          ; <i32> [#uses=1]
        %tmp.3 = add i32 %tmp.2, -32768         ; <i32> [#uses=1]
        ret i32 %tmp.3
; CHECK: @test2
; CHECK: %sext = shl i32 %x, 16
; CHECK: %tmp.3 = ashr exact i32 %sext, 16
; CHECK: ret i32 %tmp.3
}

define i32 @test3(i16 %P) {
        %tmp.1 = zext i16 %P to i32             ; <i32> [#uses=1]
        %tmp.4 = xor i32 %tmp.1, 32768          ; <i32> [#uses=1]
        %tmp.5 = add i32 %tmp.4, -32768         ; <i32> [#uses=1]
        ret i32 %tmp.5
; CHECK: @test3
; CHECK: %tmp.5 = sext i16 %P to i32
; CHECK: ret i32 %tmp.5
}

define i32 @test4(i16 %P) {
        %tmp.1 = zext i16 %P to i32             ; <i32> [#uses=1]
        %tmp.4 = xor i32 %tmp.1, 32768          ; <i32> [#uses=1]
        %tmp.5 = add i32 %tmp.4, -32768         ; <i32> [#uses=1]
        ret i32 %tmp.5
; CHECK: @test4
; CHECK: %tmp.5 = sext i16 %P to i32
; CHECK: ret i32 %tmp.5
}

define i32 @test5(i32 %x) {
        %tmp.1 = and i32 %x, 255                ; <i32> [#uses=1]
        %tmp.2 = xor i32 %tmp.1, 128            ; <i32> [#uses=1]
        %tmp.3 = add i32 %tmp.2, -128           ; <i32> [#uses=1]
        ret i32 %tmp.3
; CHECK: @test5
; CHECK: %sext = shl i32 %x, 24
; CHECK: %tmp.3 = ashr exact i32 %sext, 24
; CHECK: ret i32 %tmp.3
}

define i32 @test6(i32 %x) {
        %tmp.2 = shl i32 %x, 16         ; <i32> [#uses=1]
        %tmp.4 = ashr i32 %tmp.2, 16            ; <i32> [#uses=1]
        ret i32 %tmp.4
; CHECK: @test6
; CHECK: %tmp.2 = shl i32 %x, 16
; CHECK: %tmp.4 = ashr exact i32 %tmp.2, 16
; CHECK: ret i32 %tmp.4
}

define i32 @test7(i16 %P) {
  %tmp.1 = zext i16 %P to i32                     ; <i32> [#uses=1]
  %sext1 = shl i32 %tmp.1, 16                     ; <i32> [#uses=1]
  %tmp.5 = ashr i32 %sext1, 16                    ; <i32> [#uses=1]
  ret i32 %tmp.5
; CHECK: @test7
; CHECK: %tmp.5 = sext i16 %P to i32
; CHECK: ret i32 %tmp.5
}

define i32 @test8(i32 %x) nounwind readnone {
entry:
  %shr = lshr i32 %x, 5                           ; <i32> [#uses=1]
  %xor = xor i32 %shr, 67108864                   ; <i32> [#uses=1]
  %sub = add i32 %xor, -67108864                  ; <i32> [#uses=1]
  ret i32 %sub
; CHECK: @test8
; CHECK: %shr = ashr i32 %x, 5
; CHECK: ret i32 %shr
}
