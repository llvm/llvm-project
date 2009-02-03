; RUN: llvm-as < %s | opt -inline | llvm-dis | not grep tail

declare void @bar(i32*)

define internal void @foo(i32* %P) {
        tail call void @bar( i32* %P )
        ret void
}

define void @caller() {
        %A = alloca i32         ; <i32*> [#uses=1]
        call void @foo( i32* %A )
        ret void
}

