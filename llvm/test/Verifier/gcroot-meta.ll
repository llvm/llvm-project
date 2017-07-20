; RUN: not llvm-as < %s > /dev/null 2>&1
; PR1633

%meta = type { i8* }
%obj = type { %meta* }

declare void @llvm.gcroot(%obj**, %meta*)

define void @f() {
entry:
	%local.obj = alloca %obj*
	%local.meta = alloca %meta
	call void @llvm.gcroot(%obj** %local.obj, %meta* %local.meta)
	
	ret void
}
