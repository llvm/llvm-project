; RUN: %lli -remote-mcjit -mcjit-remote-process=lli-child-target%exeext %s > /dev/null

define i32 @bar() nounwind {
	ret i32 0
}

define i32 @main() nounwind {
	%r = call i32 @bar( )		; <i32> [#uses=1]
	ret i32 %r
}
