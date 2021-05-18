; RUN: opt -passes=verify -S < %s | FileCheck %s

; CHECK: @load
define i32 @load(ptr %a) {
    %i = load i32, ptr %a
    ret i32 %i
}

; CHECK: @store
define void @store(ptr %a, i32 %i) {
    store i32 %i, ptr %a
    ret void
}
