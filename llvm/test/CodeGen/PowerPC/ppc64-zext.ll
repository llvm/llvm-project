; RUN: llc < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux"

define i64 @fun(i32 %arg32) nounwind {
entry:
; CHECK: rldicl {{[0-9]+}}, {{[0-9]+}}, 0, 32
  %o = zext i32 %arg32 to i64
  ret i64 %o
}

