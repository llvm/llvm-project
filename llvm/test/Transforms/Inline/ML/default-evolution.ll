; RUN: opt -passes='default<O3>' \
; RUN:   -S -enable-ml-inliner=evolution < %s 2>&1 | FileCheck %s
; RUN: opt -passes='default<O3>' \
; RUN:   -S -enable-ml-inliner=default < %s 2>&1 | FileCheck %s

declare i32 @f1()

define i32 @f2() {
    ret i32 1
}

define i32 @f3() {
    %a = call i32 @f1()
    %b = call i32 @f2()
    %c = add i32 %a, %b
    ret i32 %c
}

; all the functions are not inlined by default
; CHECK-LABEL: @f1
; CHECK-LABEL: @f2
; CHECK-LABEL: @f3
; default inlining policy may inline @f2
; CHECK-LABEL: @f1
; CHECK-NOT-LABEL: @f2
; CHECK-LABEL: @f3