; RUN: opt < %s -passes=globalopt -S | FileCheck %s

@G1 = internal global i32 123            ; <ptr> [#uses=1]
@A1 = internal alias i32, ptr @G1

; CHECK-NOT: @G1
; CHECK: @G2
; CHECK-NOT: @G3

; CHECK-NOT: @A1

define void @foo1() {
; CHECK: define void @foo
; CHECK-NEXT: ret
        store i32 1, ptr @G1
        ret void
}

@G2 = linkonce_odr constant i32 42

define void @foo2() {
; CHECK-LABEL: define void @foo2(
; CHECK-NEXT: store
        store i32 1, ptr @G2
        ret void
}

@G3 = linkonce_odr constant i32 42
