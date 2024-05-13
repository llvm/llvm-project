; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; This is a harder case to delete as the GEP has a variable index.

; CHECK-NOT: internal
@G = internal global [4 x i32] zeroinitializer

define void @foo(i32 %X) {
	%Ptr = getelementptr [4 x i32], ptr @G, i32 0, i32 %X
	store i32 1, ptr %Ptr
	ret void
}
