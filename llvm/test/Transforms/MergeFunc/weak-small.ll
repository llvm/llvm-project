; RUN: opt -mergefunc -S < %s | FileCheck %s

; Weak functions too small for merging to be profitable

; CHECK: define weak i32 @foo(i8*, i32)
; CHECK-NEXT: ret i32 %1
; CHECK: define weak i32 @bar(i8*, i32)
; CHECK-NEXT: ret i32 %1

define weak i32 @foo(i8*, i32) #0 {
    ret i32 %1
}

define weak i32 @bar(i8*, i32) #0 {
    ret i32 %1
}
