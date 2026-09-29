; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; CHECK-NOT: %G

@G = internal global i32 0              ; <ptr> [#uses=1]
@H = internal global { ptr } { ptr @G }               ; <ptr> [#uses=1]

define i32 @loadg() {
        %G = load ptr, ptr @H              ; <ptr> [#uses=1]
        %GV = load i32, ptr %G              ; <i32> [#uses=1]
        ret i32 %GV
}
