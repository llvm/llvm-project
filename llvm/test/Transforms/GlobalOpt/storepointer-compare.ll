; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; CHECK: call void @Actual

; Check that a comparison does not prevent an indirect call from being made 
; direct.  The global will still remain, but indirect call elim is still good.

@G = internal global ptr null              ; <ptr> [#uses=2]

define internal void @Actual() {
        ret void
}

define void @init() {
        store ptr @Actual, ptr @G
        ret void
}

define void @doit() {
        %FP = load ptr, ptr @G         ; <ptr> [#uses=2]
        %CC = icmp eq ptr %FP, null                ; <i1> [#uses=1]
        br i1 %CC, label %isNull, label %DoCall

DoCall:         ; preds = %0
        call void %FP( )
        ret void

isNull:         ; preds = %0
        ret void
}
