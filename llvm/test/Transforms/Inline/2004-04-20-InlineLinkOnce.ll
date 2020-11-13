; RUN: opt < %s -inline -prune-eh -disable-output -enable-new-pm=0

define linkonce void @caller() {
        call void @callee( )
        ret void
}

define linkonce void @callee() {
        ret void
}

