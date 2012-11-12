; RUN: llc < %s | grep p-92
; PR3481
; The offset should print as -92, not +17179869092

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
@p = common global [10 x i32] zeroinitializer, align 4          ; <[10 x i32]*>
@g = global [1 x i32*] [ i32* bitcast (i8* getelementptr (i8* bitcast
([10 x i32]* @p to i8*), i64 17179869092) to i32*) ], align 4 
