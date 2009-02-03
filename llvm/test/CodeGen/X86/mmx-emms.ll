; RUN: llvm-as < %s | llc -march=x86 -mattr=+mmx | grep emms
define void @foo() {
entry:
	call void @llvm.x86.mmx.emms( )
	br label %return

return:		; preds = %entry
	ret void
}

declare void @llvm.x86.mmx.emms()
