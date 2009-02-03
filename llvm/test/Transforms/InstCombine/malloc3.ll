; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {malloc.*struct.foo} | count 2
; PR1728

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"
        %struct.foo = type { %struct.foo*, [10 x i32] }
@.str = internal constant [21 x i8] c"tmp = %p, next = %p\0A\00"                ; <[21 x i8]*> [#uses=1]

define i32 @main() {
entry:
        %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
        %tmp1 = malloc i8, i32 44               ; <i8*> [#uses=1]
        %tmp12 = bitcast i8* %tmp1 to %struct.foo*              ; <%struct.foo*> [#uses=3]
        %tmp3 = malloc i8, i32 44               ; <i8*> [#uses=1]
        %tmp34 = bitcast i8* %tmp3 to %struct.foo*              ; <%struct.foo*> [#uses=1]
        %tmp6 = getelementptr %struct.foo* %tmp12, i32 0, i32 0         ; <%struct.foo**> [#uses=1]
        store %struct.foo* %tmp34, %struct.foo** %tmp6, align 4
        %tmp8 = getelementptr %struct.foo* %tmp12, i32 0, i32 0         ; <%struct.foo**> [#uses=1]
        %tmp9 = load %struct.foo** %tmp8, align 4               ; <%struct.foo*> [#uses=1]
        %tmp10 = getelementptr [21 x i8]* @.str, i32 0, i32 0           ; <i8*> [#uses=1]
        %tmp13 = call i32 (i8*, ...)* @printf( i8* %tmp10, %struct.foo* %tmp12, %struct.foo* %tmp9 )            ; <i32> [#uses=0]
        ret i32 undef
}

declare i32 @printf(i8*, ...)

