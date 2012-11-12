; RUN: opt -S -instcombine < %s | FileCheck %s

; rdar://12182093

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; CHECK: @udiv400
; CHECK: udiv i32 %x, 400
; CHECK: ret
define i32 @udiv400(i32 %x) {
entry:
  %div = lshr i32 %x, 2
  %div1 = udiv i32 %div, 100
  ret i32 %div1
}


; CHECK: @udiv400_no
; CHECK: ashr
; CHECK: div
; CHECK: ret
define i32 @udiv400_no(i32 %x) {
entry:
  %div = ashr i32 %x, 2
  %div1 = udiv i32 %div, 100
  ret i32 %div1
}

; CHECK: @sdiv400_yes
; CHECK: udiv i32 %x, 400
; CHECK: ret
define i32 @sdiv400_yes(i32 %x) {
entry:
  %div = lshr i32 %x, 2
  ; The sign bits of both operands are zero (i.e. we can prove they are
  ; unsigned inputs), turn this into a udiv.
  ; Next, optimize this just like sdiv.
  %div1 = sdiv i32 %div, 100
  ret i32 %div1
}


; CHECK: @udiv_i80
; CHECK: udiv i80 %x, 400
; CHECK: ret
define i80 @udiv_i80(i80 %x) {
  %div = lshr i80 %x, 2
  %div1 = udiv i80 %div, 100
  ret i80 %div1
}

define i32 @no_crash_notconst_udiv(i32 %x, i32 %notconst) {
  %div = lshr i32 %x, %notconst
  %div1 = udiv i32 %div, 100
  ret i32 %div1
}
