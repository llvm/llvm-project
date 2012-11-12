; RUN: opt < %s -inline -disable-output

; Inlining the first call caused the inliner function to delete the second
; call.  Then the inliner tries to inline the second call, which no longer
; exists.

define internal void @Callee1() {
        unreachable
}

define void @Callee2() {
        ret void
}

define void @caller() {
        call void @Callee1( )
        call void @Callee2( )
        ret void
}

