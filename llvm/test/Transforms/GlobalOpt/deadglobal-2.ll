; RUN: opt < %s -globalopt -S | not grep internal

; This is a harder case to delete as the GEP has a variable index.

@G = internal global [4 x i32] zeroinitializer

define void @foo(i32 %X) {
	%Ptr = getelementptr [4 x i32]* @G, i32 0, i32 %X
	store i32 1, i32* %Ptr
	ret void
}
