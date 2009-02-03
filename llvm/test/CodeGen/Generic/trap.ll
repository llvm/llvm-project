; RUN: llvm-as < %s | llc
define i32 @test() noreturn nounwind  {
entry:
	tail call void @llvm.trap( )
	unreachable
}

declare void @llvm.trap() nounwind 

