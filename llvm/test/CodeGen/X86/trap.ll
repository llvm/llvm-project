; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah | grep ud2
define i32 @test() noreturn nounwind  {
entry:
	tail call void @llvm.trap( )
	unreachable
}

declare void @llvm.trap() nounwind 

