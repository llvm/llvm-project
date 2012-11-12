; RUN: opt %s -instcombine -S | FileCheck %s
; PR5438

; TODO: This should also optimize down.
;define i32 @test1(i32 %a, i32 %b) nounwind readnone {
;entry:
;        %0 = icmp sgt i32 %a, -1        ; <i1> [#uses=1]
;        %1 = icmp slt i32 %b, 0         ; <i1> [#uses=1]
;        %2 = xor i1 %1, %0              ; <i1> [#uses=1]
;        %3 = zext i1 %2 to i32          ; <i32> [#uses=1]
;        ret i32 %3
;}

; TODO: This optimizes partially but not all the way.
;define i32 @test2(i32 %a, i32 %b) nounwind readnone {
;entry:
;        %0 = and i32 %a, 8            ;<i32>  [#uses=1]
;        %1 = and i32 %b, 8            ;<i32>  [#uses=1]
;        %2 = icmp eq i32 %0, %1         ;<i1>  [#uses=1]
;        %3 = zext i1 %2 to i32          ;<i32>  [#uses=1]
;        ret i32 %3
;}

define i32 @test3(i32 %a, i32 %b) nounwind readnone {
; CHECK: @test3
entry:
; CHECK: xor i32 %a, %b
; CHECK: lshr i32 %0, 31
; CHECK: xor i32 %1, 1
        %0 = lshr i32 %a, 31            ; <i32> [#uses=1]
        %1 = lshr i32 %b, 31            ; <i32> [#uses=1]
        %2 = icmp eq i32 %0, %1         ; <i1> [#uses=1]
        %3 = zext i1 %2 to i32          ; <i32> [#uses=1]
        ret i32 %3
; CHECK-NOT: icmp
; CHECK-NOT: zext
; CHECK: ret i32 %2
}

; Variation on @test3: checking the 2nd bit in a situation where the 5th bit
; is one, not zero.
define i32 @test3i(i32 %a, i32 %b) nounwind readnone {
; CHECK: @test3i
entry:
; CHECK: xor i32 %a, %b
; CHECK: lshr i32 %0, 31
; CHECK: xor i32 %1, 1
        %0 = lshr i32 %a, 29            ; <i32> [#uses=1]
        %1 = lshr i32 %b, 29            ; <i32> [#uses=1]
        %2 = or i32 %0, 35
        %3 = or i32 %1, 35
        %4 = icmp eq i32 %2, %3         ; <i1> [#uses=1]
        %5 = zext i1 %4 to i32          ; <i32> [#uses=1]
        ret i32 %5
; CHECK-NOT: icmp
; CHECK-NOT: zext
; CHECK: ret i32 %2
}
