; RUN: llc < %s -mtriple=ppc64--

define ptr @foo(i32 %n) {
        %A = alloca i32, i32 %n         ; <ptr> [#uses=1]
        ret ptr %A
}

