; RUN: llvm-as < %s | opt -dse | llvm-dis | not grep store

%struct.x = type { i32, i32, i32, i32 }

define i32 @foo(%struct.x* byval  %a) nounwind  {
entry:
	%tmp2 = getelementptr %struct.x* %a, i32 0, i32 0
	store i32 1, i32* %tmp2, align 4
	ret i32 1
}
