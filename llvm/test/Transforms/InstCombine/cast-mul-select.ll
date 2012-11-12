; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32"

define i32 @mul(i32 %x, i32 %y) {
  %A = trunc i32 %x to i8
  %B = trunc i32 %y to i8
  %C = mul i8 %A, %B
  %D = zext i8 %C to i32
  ret i32 %D
; CHECK: %C = mul i32 %x, %y
; CHECK: %D = and i32 %C, 255
; CHECK: ret i32 %D
}

define i32 @select1(i1 %cond, i32 %x, i32 %y, i32 %z) {
  %A = trunc i32 %x to i8
  %B = trunc i32 %y to i8
  %C = trunc i32 %z to i8
  %D = add i8 %A, %B
  %E = select i1 %cond, i8 %C, i8 %D
  %F = zext i8 %E to i32
  ret i32 %F
; CHECK: %D = add i32 %x, %y
; CHECK: %E = select i1 %cond, i32 %z, i32 %D
; CHECK: %F = and i32 %E, 255
; CHECK: ret i32 %F
}

define i8 @select2(i1 %cond, i8 %x, i8 %y, i8 %z) {
  %A = zext i8 %x to i32
  %B = zext i8 %y to i32
  %C = zext i8 %z to i32
  %D = add i32 %A, %B
  %E = select i1 %cond, i32 %C, i32 %D
  %F = trunc i32 %E to i8
  ret i8 %F
; CHECK: %D = add i8 %x, %y
; CHECK: %E = select i1 %cond, i8 %z, i8 %D
; CHECK: ret i8 %E
}
