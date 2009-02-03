; RUN: not llvm-as < %s >& /dev/null
; PR2711
	%s = type opaque
declare void @h(%s* byval %num)
