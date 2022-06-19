; RUN: opt -passes='default<O3>,print<inline-advisor>' -training-log=/dev/null \
; RUN:   -S -enable-ml-inliner=development -keep-inline-advisor-for-printing < %s 2>&1 | FileCheck %s
; REQUIRES: have_tf_api
;
; CHECK: [MLInlineAdvisor] Nodes: 3 Edges: 1

declare i32 @f1()

define i32 @f2() {
    ret i32 1
}

define i32 @f3() noinline {
    ret i32 2
}

define i32 @f4() {
    %a = call i32 @f1()
    %b = call i32 @f2()
    %c = call i32 @f3()
    %d = add i32 %a, %b
    %e = add i32 %d, %c
    ret i32 %e
}