; RUN: not llvm-as < %s >& /dev/null
; PR1633

%meta = type { i8* }
%obj = type { %meta* }

declare %obj* @llvm.gcread(%obj*, %obj*)

define %obj* @f() {
entry:
	%x = call %obj* @llvm.gcread(%obj* null, %obj* null)
	ret %obj* %x
}
