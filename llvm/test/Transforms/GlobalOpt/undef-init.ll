; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; CHECK-NOT: store

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [ { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__I__Z3foov, i8* null } ]          ; <[1 x { i32, void ()*, i8* }]*> [#uses=0]
@X.0 = internal global i32 undef                ; <i32*> [#uses=2]

define i32 @_Z3foov() {
entry:
        %tmp.1 = load i32, i32* @X.0         ; <i32> [#uses=1]
        ret i32 %tmp.1
}

define internal void @_GLOBAL__I__Z3foov() {
entry:
        store i32 1, i32* @X.0
        ret void
}
