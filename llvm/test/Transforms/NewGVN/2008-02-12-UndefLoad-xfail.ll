; XFAIL: *
; RUN: opt < %s -passes=newgvn -S | FileCheck %s
; PR1996

%struct.anon = type { i32, i8, i8, i8, i8 }

define i32 @a() {
entry:
        %c = alloca %struct.anon                ; <ptr> [#uses=2]
        %tmp = getelementptr %struct.anon, ptr %c, i32 0, i32 0             ; <ptr> [#uses=1]
        %tmp1 = getelementptr i32, ptr %tmp, i32 1          ; <ptr> [#uses=2]
        %tmp2 = load i32, ptr %tmp1, align 4                ; <i32> [#uses=1]
; CHECK-NOT: load
        %tmp3 = or i32 %tmp2, 11                ; <i32> [#uses=1]
        %tmp4 = and i32 %tmp3, -21              ; <i32> [#uses=1]
        store i32 %tmp4, ptr %tmp1, align 4
        %call = call i32 (...) @x( ptr %c )          ; <i32> [#uses=0]
        ret i32 undef
}


declare i32 @x(...)
