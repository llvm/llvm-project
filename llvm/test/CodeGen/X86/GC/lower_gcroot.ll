; RUN: llc -mtriple=x86_64 < %s

	%Env = type ptr

define void @.main(%Env) gc "shadow-stack" {
	%Root = alloca %Env
	call void @llvm.gcroot( ptr %Root, %Env null )
	unreachable
}

declare void @llvm.gcroot(ptr, %Env)
