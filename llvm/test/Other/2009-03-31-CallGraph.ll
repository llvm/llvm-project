; RUN: opt < %s -inline -prune-eh -disable-output -enable-new-pm=0
define void @f2() personality i32 (...)* @__gxx_personality_v0 {
    invoke void @f6()
        to label %ok1 unwind label %lpad1

ok1:
    ret void

lpad1:
    landingpad {i8*, i32}
            cleanup
    invoke void @f4()
        to label %ok2 unwind label %lpad2

ok2:
    call void @f8()
    unreachable

lpad2:
    %exn = landingpad {i8*, i32}
            cleanup
    unreachable
}

declare void @f3()

define void @f4() {
    call void @f3()
    ret void
}

declare void @f6() nounwind

declare void @f8()

declare i32 @__gxx_personality_v0(...)
