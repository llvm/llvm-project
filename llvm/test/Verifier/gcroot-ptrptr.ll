; RUN: not llvm-as < %s >& /dev/null
; PR1633

%meta = type { i8* }
%obj = type { %meta* }

declare void @llvm.gcroot(%obj*, %meta*)

define void @f() {
entry:
	%local.obj = alloca %obj
	call void @llvm.gcroot(%obj* %local.obj, %meta* null)
	ret void
}
