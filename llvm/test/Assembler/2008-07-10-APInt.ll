; RUN: llvm-as < %s | llvm-dis
; PR2538
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"

define i128 @a() {
        ret i128 18446744073709551616
}

