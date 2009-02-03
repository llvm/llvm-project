; RUN: llvm-as < %s | llc | grep {addl	\$3, (%eax)} | count 4
; PR2182

target datalayout =
"e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"
@x = weak global i32 0          ; <i32*> [#uses=8]

define void @loop_2() nounwind  {
entry:
        %tmp = volatile load i32* @x, align 4           ; <i32> [#uses=1]
        %tmp1 = add i32 %tmp, 3         ; <i32> [#uses=1]
        volatile store i32 %tmp1, i32* @x, align 4
        %tmp.1 = volatile load i32* @x, align 4         ; <i32> [#uses=1]
        %tmp1.1 = add i32 %tmp.1, 3             ; <i32> [#uses=1]
        volatile store i32 %tmp1.1, i32* @x, align 4
        %tmp.2 = volatile load i32* @x, align 4         ; <i32> [#uses=1]
        %tmp1.2 = add i32 %tmp.2, 3             ; <i32> [#uses=1]
        volatile store i32 %tmp1.2, i32* @x, align 4
        %tmp.3 = volatile load i32* @x, align 4         ; <i32> [#uses=1]
        %tmp1.3 = add i32 %tmp.3, 3             ; <i32> [#uses=1]
        volatile store i32 %tmp1.3, i32* @x, align 4
        ret void
}
