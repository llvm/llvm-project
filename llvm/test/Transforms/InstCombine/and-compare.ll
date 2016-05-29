; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Should be optimized to one and.
define i1 @test1(i32 %a, i32 %b) {
; CHECK-LABEL: @test1(
; CHECK-NEXT: %1 = xor i32 %a, %b
; CHECK-NEXT: %2 = and i32 %1, 65280
; CHECK-NEXT: %tmp = icmp ne i32 %2, 0
; CHECK-NEXT: ret i1 %tmp
        %tmp1 = and i32 %a, 65280               ; <i32> [#uses=1]
        %tmp3 = and i32 %b, 65280               ; <i32> [#uses=1]
        %tmp = icmp ne i32 %tmp1, %tmp3         ; <i1> [#uses=1]
        ret i1 %tmp
}

define zeroext i1 @test2(i64 %A) {
; CHECK-LABEL: @test2(
; CHECK-NEXT: %[[trunc:.*]] = trunc i64 %A to i8
; CHECK-NEXT: %[[icmp:.*]] = icmp sgt i8 %[[trunc]], -1
; CHECK-NEXT: ret i1 %[[icmp]]
  %and = and i64 %A, 128
  %cmp = icmp eq i64 %and, 0
  ret i1 %cmp
}

define zeroext i1 @test3(i64 %A) {
; CHECK-LABEL: @test3(
; CHECK-NEXT: %[[trunc:.*]] = trunc i64 %A to i8
; CHECK-NEXT: %[[icmp:.*]] = icmp slt i8 %[[trunc]], 0
; CHECK-NEXT: ret i1 %[[icmp]]
  %and = and i64 %A, 128
  %cmp = icmp ne i64 %and, 0
  ret i1 %cmp
}
