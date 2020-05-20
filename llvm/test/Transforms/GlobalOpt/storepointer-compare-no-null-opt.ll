; RUN: opt < %s -globalopt -S | FileCheck %s
; CHECK: global

@G = internal global void ()* null              ; <void ()**> [#uses=2]

define internal void @Actual() {
; CHECK-LABEL: Actual(
        ret void
}

define void @init() {
; CHECK-LABEL: init(
; CHECK: store void ()* @Actual, void ()** @G
        store void ()* @Actual, void ()** @G
        ret void
}

define void @doit() #0 {
; CHECK-LABEL: doit(
        %FP = load void ()*, void ()** @G         ; <void ()*> [#uses=2]
; CHECK: %FP = load void ()*, void ()** @G
        %CC = icmp eq void ()* %FP, null                ; <i1> [#uses=1]
; CHECK: %CC = icmp eq void ()* %FP, null
        br i1 %CC, label %isNull, label %DoCall
; CHECK: br i1 %CC, label %isNull, label %DoCall

DoCall:         ; preds = %0
; CHECK: DoCall:
; CHECK: call void %FP()
; CHECK: ret void
        call void %FP( )
        ret void

isNull:         ; preds = %0
; CHECK: isNull:
; CHECK: ret void
        ret void
}

attributes #0 = { "null-pointer-is-valid"="true" }
