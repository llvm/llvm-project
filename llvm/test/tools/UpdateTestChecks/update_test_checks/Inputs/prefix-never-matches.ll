; RUN: opt -O0 -S < %s  | FileCheck %s -check-prefix=A
; RUN: opt -O3 -S < %s  | FileCheck %s -check-prefix=A

define i32 @foo(i32 %i) {
    %r = add i32 1, 1
    ret i32 %r
}