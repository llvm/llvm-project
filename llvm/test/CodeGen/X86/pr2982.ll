; RUN: llc < %s
; PR2982

target datalayout =
"e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.5"
@g_279 = external global i32            ; <ptr> [#uses=1]
@g_265 = external global i32            ; <ptr> [#uses=1]
@g_3 = external global i8               ; <ptr> [#uses=1]

declare i32 @rshift_u_u(...)

define void @bar() nounwind {
entry:
        %0 = load i32, ptr @g_279, align 4          ; <i32> [#uses=1]
        %1 = shl i32 %0, 1              ; <i32> [#uses=1]
        %2 = and i32 %1, 2              ; <i32> [#uses=1]
        %3 = load i32, ptr @g_265, align 4          ; <i32> [#uses=1]
        %4 = load i8, ptr @g_3, align 1             ; <i8> [#uses=1]
        %5 = sext i8 %4 to i32          ; <i32> [#uses=1]
        %6 = add i32 %2, %3             ; <i32> [#uses=1]
        %7 = add i32 %6, %5             ; <i32> [#uses=1]
        %8 = tail call i32 (...) @rshift_u_u(i32 %7, i32 0) nounwind          
; <i32> [#uses=0]
        ret void
}
