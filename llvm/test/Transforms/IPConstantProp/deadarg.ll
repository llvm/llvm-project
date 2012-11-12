; RUN: opt < %s -ipconstprop -disable-output
define internal void @foo(i32 %X) {
        call void @foo( i32 %X )
        ret void
}

