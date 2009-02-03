; RUN: llvm-as < %s | opt -basicaa -gvn -dce | llvm-dis | grep tmp7

        %struct.A = type { i32 }
        %struct.B = type { %struct.A }
@a = global %struct.B zeroinitializer           ; <%struct.B*> [#uses=2]

define i32 @_Z3fooP1A(%struct.A* %b) {
entry:
        store i32 1, i32* getelementptr (%struct.B* @a, i32 0, i32 0, i32 0), align 8
        %tmp4 = getelementptr %struct.A* %b, i32 0, i32 0               ;<i32*> [#uses=1]
        store i32 0, i32* %tmp4, align 4
        %tmp7 = load i32* getelementptr (%struct.B* @a, i32 0, i32 0, i32 0), align 8           ; <i32> [#uses=1]
        ret i32 %tmp7
}
