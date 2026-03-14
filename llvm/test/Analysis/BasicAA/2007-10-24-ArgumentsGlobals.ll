; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn,dce -S | FileCheck %s

        %struct.A = type { i32 }
        %struct.B = type { %struct.A }
@a = global %struct.B zeroinitializer           ; <ptr> [#uses=2]

define i32 @_Z3fooP1A(ptr %b) {
; CHECK: %tmp7 = load
; CHECK: ret i32 %tmp7
entry:
        store i32 1, ptr @a, align 8
        %tmp4 = getelementptr %struct.A, ptr %b, i32 0, i32 0               ;<ptr> [#uses=1]
        store i32 0, ptr %tmp4, align 4
        %tmp7 = load i32, ptr @a, align 8           ; <i32> [#uses=1]
        ret i32 %tmp7
}
