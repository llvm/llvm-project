; RUN: llvm-as < %s | opt -sccp | llvm-dis | not grep {ret i32 undef}
; PR2358
target datalayout =
"e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-pc-linux-gnu"

define i32 @x(i32 %b) {
entry:
 %val = call i32 @llvm.cttz.i32(i32 undef)
 ret i32 %val
}

declare i32 @llvm.cttz.i32(i32)

