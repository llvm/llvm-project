; RUN: opt < %s -passes=loop-deletion -S | grep switch
; PR 1564

define fastcc void @out() {
    start:
            br label %loop
    unreachable:
            unreachable
    loop:
            switch i32 0, label %unreachable [
                     i32 0, label %loop
            ]
}
