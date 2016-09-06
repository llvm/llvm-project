; RUN: opt < %s -globalopt -S | FileCheck %s
; CHECK-NOT: global

@G = internal global void ()* null              ; <void ()**> [#uses=2]

define internal void @Actual() {
        ret void
}

define void @init() {
        store void ()* @Actual, void ()** @G
        ret void
}

define void @doit() {
        %FP = load void ()*, void ()** @G         ; <void ()*> [#uses=1]
        call void %FP( )
        ret void
}
