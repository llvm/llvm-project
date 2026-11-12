; RUN: opt < %s -passes=lint -disable-output 2>&1 | FileCheck %s

define void @test() {
entry:
    %tmp = alloca i32, i32 -5
    ret void
}

; CHECK: Undefined behavior: Static alloca used with a negative number
