; RUN: llvm-as < %s | llc -march=x86-64 | grep movq | grep rbp

define i64* @stack_end_address() nounwind  {
entry:
	tail call i8* @llvm.frameaddress( i32 0 )
	bitcast i8* %0 to i64*
	ret i64* %1
}

declare i8* @llvm.frameaddress(i32) nounwind readnone 
