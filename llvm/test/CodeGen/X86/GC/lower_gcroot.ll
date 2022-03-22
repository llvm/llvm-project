; RUN: llc -mtriple=x86_64 < %s

	%Env = type i8*

define void @.main(%Env) gc "shadow-stack" {
	%Root = alloca %Env
	call void @llvm.gcroot( %Env* %Root, %Env null )
	unreachable
}

declare void @llvm.gcroot(%Env*, %Env)
