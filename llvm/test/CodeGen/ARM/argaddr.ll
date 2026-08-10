; RUN: llc -mtriple=arm-eabi %s -o /dev/null

define void @f(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
entry:
        %a_addr = alloca i32            ; <ptr> [#uses=2]
        %b_addr = alloca i32            ; <ptr> [#uses=2]
        %c_addr = alloca i32            ; <ptr> [#uses=2]
        %d_addr = alloca i32            ; <ptr> [#uses=2]
        %e_addr = alloca i32            ; <ptr> [#uses=2]
        store i32 %a, ptr %a_addr
        store i32 %b, ptr %b_addr
        store i32 %c, ptr %c_addr
        store i32 %d, ptr %d_addr
        store i32 %e, ptr %e_addr
        call void @g( ptr %a_addr, ptr %b_addr, ptr %c_addr, ptr %d_addr, ptr %e_addr )
        ret void
}

declare void @g(ptr, ptr, ptr, ptr, ptr)
