; RUN: not llvm-as < %s >& /dev/null
; PR1633

%meta = type { i8* }
%obj = type { %meta* }

declare void @llvm.gcroot(%obj**, %meta*)

define void @f() {
entry:
	call void @llvm.gcroot(%obj** null, %meta* null)
	
	ret void
}
