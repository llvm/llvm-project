; RUN: opt < %s -inline -prune-eh -disable-output

define linkonce void @caller() {
        call void @callee( )
        ret void
}

define linkonce void @callee() {
        ret void
}

