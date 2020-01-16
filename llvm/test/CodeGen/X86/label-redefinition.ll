; PR7054
; RUN: not --crash llc %s -o - 2>&1 | grep "'_foo' label emitted multiple times to assembly"
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin10.0.0"

define i32 @"\01_foo"() {
  unreachable
}

define i32 @foo() {
entry:
  unreachable
}

declare i32 @xstat64(i32, i8*, i8*)
