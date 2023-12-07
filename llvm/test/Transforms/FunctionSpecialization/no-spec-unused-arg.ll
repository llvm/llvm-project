; RUN: opt -S --passes="ipsccp<func-spec>" -force-specialization -funcspec-for-literal-constant < %s | FileCheck %s
define internal i32 @f(i32 %x, i32 %y) noinline {
    ret i32 %x
}

define i32 @g0() {
    %r = call i32 @f(i32 1, i32 1)
    ret i32 %r
}

define i32 @g1() {
    %r = call i32 @f(i32 1, i32 2)
    ret i32 %r
}

; Check that there are no specialisation of `f`: first parameter is deduced
; to be a constant without the need for function specialisation and
; the second parameter is unused.

;  CHECK-NOT: @f.
