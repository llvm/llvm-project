; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; CHECK-NOT: global

@G = internal global ptr null              ; <ptr> [#uses=2]

define internal void @Actual() {
        ret void
}

define void @init() {
        store ptr @Actual, ptr @G
        ret void
}

define void @doit() {
        %FP = load ptr, ptr @G         ; <ptr> [#uses=1]
        call void %FP( )
        ret void
}
