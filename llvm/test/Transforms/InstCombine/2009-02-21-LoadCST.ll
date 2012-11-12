; RUN: opt < %s -instcombine -S | grep "ret i32 3679669"
; PR3595

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

@.str1 = internal constant [4 x i8] c"\B5%8\00"

define i32 @test() {
  %rhsv = load i32* bitcast ([4 x i8]* @.str1 to i32*), align 1
  ret i32 %rhsv
}
