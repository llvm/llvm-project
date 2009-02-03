; RUN: llvm-as < %s | opt -gvn | llvm-dis | not grep load
; PR1996

%struct.anon = type { i32, i8, i8, i8, i8 }

define i32 @a() {
entry:
        %c = alloca %struct.anon                ; <%struct.anon*> [#uses=2]
        %tmp = getelementptr %struct.anon* %c, i32 0, i32 0             ; <i32*> [#uses=1]
        %tmp1 = getelementptr i32* %tmp, i32 1          ; <i32*> [#uses=2]
        %tmp2 = load i32* %tmp1, align 4                ; <i32> [#uses=1]
        %tmp3 = or i32 %tmp2, 11                ; <i32> [#uses=1]
        %tmp4 = and i32 %tmp3, -21              ; <i32> [#uses=1]
        store i32 %tmp4, i32* %tmp1, align 4
        %call = call i32 (...)* @x( %struct.anon* %c )          ; <i32> [#uses=0]
        ret i32 undef
}


declare i32 @x(...)
