; RUN: opt < %s -passes=globalopt -S | FileCheck %s

@G = internal global ptr null              ; <ptr> [#uses=2]
; CHECK: global

define internal void @Actual() {
; CHECK-LABEL: Actual(
        ret void
}

define void @init() {
; CHECK-LABEL: init(
; CHECK:  store ptr @Actual, ptr @G
        store ptr @Actual, ptr @G
        ret void
}

define void @doit() #0 {
; CHECK-LABEL: doit(
; CHECK: %FP = load ptr, ptr @G
; CHECK: call void %FP()
        %FP = load ptr, ptr @G         ; <ptr> [#uses=1]
        call void %FP( )
        ret void
}

attributes #0 = { null_pointer_is_valid }
