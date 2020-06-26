; RUN: opt < %s -basic-aa -gvn -dce -S | FileCheck %s

        %struct.A = type { i32 }
        %struct.B = type { %struct.A }
@a = global %struct.B zeroinitializer           ; <%struct.B*> [#uses=2]

define i32 @_Z3fooP1A(%struct.A* %b) {
; CHECK: %tmp7 = load
; CHECK: ret i32 %tmp7
entry:
        store i32 1, i32* getelementptr (%struct.B, %struct.B* @a, i32 0, i32 0, i32 0), align 8
        %tmp4 = getelementptr %struct.A, %struct.A* %b, i32 0, i32 0               ;<i32*> [#uses=1]
        store i32 0, i32* %tmp4, align 4
        %tmp7 = load i32, i32* getelementptr (%struct.B, %struct.B* @a, i32 0, i32 0, i32 0), align 8           ; <i32> [#uses=1]
        ret i32 %tmp7
}
