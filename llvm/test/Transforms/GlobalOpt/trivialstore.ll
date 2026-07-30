; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; CHECK-NOT: G

@G = internal global i32 17             ; <ptr> [#uses=3]

define void @foo() {
        store i32 17, ptr @G
        ret void
}

define i32 @bar() {
        %X = load i32, ptr @G               ; <i32> [#uses=1]
        ret i32 %X
}

define internal void @dead() {
        store i32 123, ptr @G
        ret void
}
