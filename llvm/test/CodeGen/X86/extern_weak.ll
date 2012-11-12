; RUN: llc < %s -mtriple=i686-apple-darwin | grep weak_reference | count 2

@Y = global i32 (i8*)* @X               ; <i32 (i8*)**> [#uses=0]

declare extern_weak i32 @X(i8*)

define void @bar() {
        tail call void (...)* @foo( )
        ret void
}

declare extern_weak void @foo(...)

