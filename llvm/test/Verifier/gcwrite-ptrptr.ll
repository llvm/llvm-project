; RUN: not llvm-as < %s > /dev/null 2>&1
; PR1633

%meta = type { i8* }
%obj = type { %meta* }

declare void @llvm.gcwrite(%obj*, %obj*, %obj*)

define void @f() {
entry:
	call void @llvm.gcwrite(%obj* null, %obj* null, %obj* null)
	ret void
}
