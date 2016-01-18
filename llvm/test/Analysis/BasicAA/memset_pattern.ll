; RUN: opt -S -basicaa -gvn < %s | FileCheck %s
; PR10872
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7"

@z = internal global i32 0, align 4
@y = internal global i32 0, align 4
@x = internal constant i32 0, align 4

; CHECK: @test
define i32 @test() nounwind uwtable ssp {
entry:
  store i32 1, i32* @z
  tail call void @memset_pattern16(i8* bitcast (i32* @y to i8*), i8* bitcast (i32* @x to i8*), i64 4) nounwind
; CHECK-NOT: load
  %l = load i32, i32* @z
; CHECK: ret i32 1
  ret i32 %l
}

declare void @memset_pattern16(i8*, i8*, i64)
