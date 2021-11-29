; RUN: opt < %s -passes=globalopt -S | FileCheck %s

@G = internal global void ()* null              ; <void ()**> [#uses=2]
; CHECK: global

define internal void @Actual() {
; CHECK-LABEL: Actual(
        ret void
}

define void @init() {
; CHECK-LABEL: init(
; CHECK:  store void ()* @Actual, void ()** @G
        store void ()* @Actual, void ()** @G
        ret void
}

define void @doit() #0 {
; CHECK-LABEL: doit(
; CHECK: %FP = load void ()*, void ()** @G
; CHECK: call void %FP()
        %FP = load void ()*, void ()** @G         ; <void ()*> [#uses=1]
        call void %FP( )
        ret void
}

attributes #0 = { null_pointer_is_valid }
