; RUN: llvm-as < %s | llc -march=x86 | grep add | not grep 16

	%struct.W = type { x86_fp80, x86_fp80 }
@B = global %struct.W { x86_fp80 0xK4001A000000000000000, x86_fp80 0xK4001C000000000000000 }, align 32
@.cpx = internal constant %struct.W { x86_fp80 0xK4001E000000000000000, x86_fp80 0xK40028000000000000000 }

define i32 @main() nounwind  {
entry:
	tail call void (i32, ...)* @bar( i32 3, %struct.W* byval  @.cpx ) nounwind 
	tail call void (i32, ...)* @baz( i32 3, %struct.W* byval  @B ) nounwind 
	ret i32 undef
}

declare void @bar(i32, ...)

declare void @baz(i32, ...)
