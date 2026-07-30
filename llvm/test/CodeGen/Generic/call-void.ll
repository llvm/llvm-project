; RUN: llc < %s

define void @foo() {
        ret void
}

define i32 @main() {
        call void @foo( )
        ret i32 0
}

