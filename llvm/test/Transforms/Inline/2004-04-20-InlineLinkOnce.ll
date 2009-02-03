; RUN: llvm-as < %s | opt -inline -prune-eh -disable-output

define linkonce void @caller() {
        call void @callee( )
        ret void
}

define linkonce void @callee() {
        ret void
}

