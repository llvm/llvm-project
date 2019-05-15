; RUN: opt -S < %s -globalopt | FileCheck %s

; Static evaluation across a @llvm.sideeffect.

; CHECK-NOT: store

declare void @llvm.sideeffect()

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [ { i32, void ()*, i8* } { i32 65535, void ()* @ctor, i8* null } ]
@G = global i32 0

define internal void @ctor() {
    store i32 1, i32* @G
    call void @llvm.sideeffect()
    ret void
}
