; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

define weak i32 @f() {
entry:
        unreachable
}

define void @g() {
entry:
        tail call void @h( )
        ret void
}

declare extern_weak void @h()

; CHECK: {{.}}weak{{.*}}f
; CHECK: {{.}}weak{{.*}}h

