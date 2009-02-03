; RUN: llvm-as < %s | llc -march=x86 | grep mov | grep ebp

define i8* @t() nounwind {
entry:
	%0 = tail call i8* @llvm.frameaddress(i32 0)
	ret i8* %0
}

declare i8* @llvm.frameaddress(i32) nounwind readnone
