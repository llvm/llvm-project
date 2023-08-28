; RUN: opt -S -passes=canon --rename-all --preserve-order < %s | FileCheck %s


; CHECK: @foo(i32 %a0)
define i32 @foo(i32) {
; CHECK: bb{{([0-9]{5})}}
entry:
    ; CHECK: %"vl{{([0-9]{5})}}(%a0, 2)"
    %a = add i32 %0, 2
    
    ; CHECK: %"op{{([0-9]{5})}}(vl{{([0-9]{5})}})"
    %b = add i32 %a, 6

    ret i32 %a
}
