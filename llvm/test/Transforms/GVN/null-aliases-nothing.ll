; RUN: opt < %s -passes=gvn -S | FileCheck %s

%t = type { i32 }
declare void @test1f(ptr)

define void @test1(ptr noalias %stuff ) {
    %before = load i32, ptr %stuff

    call void @test1f(ptr null)

    %after = load i32, ptr %stuff ; <--- This should be a dead load
    %sum = add i32 %before, %after

    store i32 %sum, ptr %stuff
    ret void
; CHECK: load
; CHECK-NOT: load
; CHECK: ret void
}
