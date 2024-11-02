; RUN: opt < %s -passes=globalopt -S | FileCheck %s

; CHECK-NOT: global
@X = internal global i32 4              ; <i32*> [#uses=1]

define i32 @foo() {
        %V = load i32, i32* @X               ; <i32> [#uses=1]
        ret i32 %V
}
