; RUN: llvm-as < %s | opt -inline -prune-eh
; PR3367

define void @f2() {
    invoke void @f6()
        to label %ok1 unwind label %lpad1

ok1:
    ret void

lpad1:
    invoke void @f4()
        to label %ok2 unwind label %lpad2

ok2:
    call void @f8()
    unreachable

lpad2:
    unreachable
}

declare void @f3()

define void @f4() {
    call void @f3()
    ret void
}

declare void @f6() nounwind

declare void @f8()
