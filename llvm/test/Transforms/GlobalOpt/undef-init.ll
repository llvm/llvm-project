; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; CHECK-NOT: store

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [ { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__I__Z3foov, ptr null } ]          ; <ptr> [#uses=0]
@X.0 = internal global i32 undef                ; <ptr> [#uses=2]

define i32 @_Z3foov() {
entry:
        %tmp.1 = load i32, ptr @X.0         ; <i32> [#uses=1]
        ret i32 %tmp.1
}

define internal void @_GLOBAL__I__Z3foov() {
entry:
        store i32 1, ptr @X.0
        ret void
}
