; RUN: opt < %s -ipconstprop -S | FileCheck %s
; Should not propagate the result of a weak function.
; PR2411

; CHECK: ret i32 %r

define weak i32 @foo() nounwind  {
entry:
        ret i32 1
}

define i32 @main() nounwind  {
entry:
        %r = call i32 @foo( ) nounwind
        ret i32 %r
}

