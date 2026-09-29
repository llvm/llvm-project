; RUN: opt -passes=mergefunc -S < %s | FileCheck %s

; Weak functions too small for merging to be profitable

; CHECK: define weak i32 @foo(ptr %0, i32 %1)
; CHECK-NEXT: ret i32 %1
; CHECK: define weak i32 @bar(ptr %0, i32 %1)
; CHECK-NEXT: ret i32 %1

define weak i32 @foo(ptr %0, i32 %1) #0 {
    ret i32 %1
}

define weak i32 @bar(ptr %0, i32 %1) #0 {
    ret i32 %1
}
