; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; CHECK: global

@G = internal global ptr null              ; <ptr> [#uses=2]

define internal void @Actual() {
; CHECK-LABEL: Actual(
        ret void
}

define void @init() {
; CHECK-LABEL: init(
; CHECK: store ptr @Actual, ptr @G
        store ptr @Actual, ptr @G
        ret void
}

define void @doit() #0 {
; CHECK-LABEL: doit(
        %FP = load ptr, ptr @G         ; <ptr> [#uses=2]
; CHECK: %FP = load ptr, ptr @G
        %CC = icmp eq ptr %FP, null                ; <i1> [#uses=1]
; CHECK: %CC = icmp eq ptr %FP, null
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

attributes #0 = { null_pointer_is_valid }
