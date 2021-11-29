; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; CHECK-NOT: %G

@G = internal global i32 0              ; <i32*> [#uses=1]
@H = internal global { i32* } { i32* @G }               ; <{ i32* }*> [#uses=1]

define i32 @loadg() {
        %G = load i32*, i32** getelementptr ({ i32* }, { i32* }* @H, i32 0, i32 0)              ; <i32*> [#uses=1]
        %GV = load i32, i32* %G              ; <i32> [#uses=1]
        ret i32 %GV
}
