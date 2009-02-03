; RUN: llvm-as < %s | llc -march=x86 | not grep mov
; RUN: llvm-as < %s | llc -march=x86-64 | not grep mov

declare void @bar()

define void @foo(i32 %i0, i32 %i1, i32 %i2, i32 %i3, i32 %i4, i32 %i5, void()* %arg) nounwind {
	call void @bar()
	call void %arg()
	ret void
}
