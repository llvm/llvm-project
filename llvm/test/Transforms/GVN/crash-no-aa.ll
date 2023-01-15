; RUN: opt -disable-basic-aa -passes=gvn -S < %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-freebsd8.0"

; PR5744
define i32 @test1(ptr %P) {
  %P2 = getelementptr {i16, i32}, ptr %P, i32 0, i32 0
  store i16 42, ptr %P2

  %P3 = getelementptr {i16, i32}, ptr %P, i32 0, i32 1
  %V = load i32, ptr %P3
  ret i32 %V
}

