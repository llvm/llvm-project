; RUN: %lli -jit-kind=mcjit -remote-mcjit -mcjit-remote-process=lli-child-target%exeext %s > /dev/null
; XFAIL: target={{.*-windows-(gnu|msvc)}}
; REQUIRES: thread_support
; UNSUPPORTED: target=powerpc64-unknown-linux-gnu
; Remove UNSUPPORTED for powerpc64-unknown-linux-gnu if problem caused by r266663 is fixed

define i32 @bar() nounwind {
	ret i32 0
}

define i32 @main() nounwind {
	%r = call i32 @bar( )		; <i32> [#uses=1]
	ret i32 %r
}
