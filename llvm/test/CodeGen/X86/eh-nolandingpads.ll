; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s
; Test that we emit functions with explicitly specified personality,
; even if no landing pads are left.

declare i32 @__my_personality_v0(...)
declare void @might_throw()

define i32 @foo() personality ptr @__my_personality_v0 {
; CHECK: .cfi_personality 3, __my_personality_v0
    call void @might_throw()
    ret i32 0
}
