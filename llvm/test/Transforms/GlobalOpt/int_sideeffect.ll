; RUN: opt -S < %s -passes=globalopt | FileCheck %s

; Static evaluation across a @llvm.sideeffect.

; CHECK-NOT: store

declare void @llvm.sideeffect()

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [ { i32, ptr, ptr } { i32 65535, ptr @ctor, ptr null } ]
@G = global i32 0

define internal void @ctor() {
    store i32 1, ptr @G
    call void @llvm.sideeffect()
    ret void
}
