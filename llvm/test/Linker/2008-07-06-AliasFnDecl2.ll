; This file is used by 2008-07-06-AliasFnDecl2.ll
; RUN: true

define void @c() nounwind  {
entry:
	call void @b( ) nounwind 
	br label %return

return:
	ret void
}

declare void @b()
