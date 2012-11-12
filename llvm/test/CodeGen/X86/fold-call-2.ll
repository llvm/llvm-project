; RUN: llc < %s -mtriple=i386-apple-darwin | grep mov | count 1

@f = external global void ()*		; <void ()**> [#uses=1]

define i32 @main() nounwind {
entry:
	load void ()** @f, align 8		; <void ()*>:0 [#uses=1]
	tail call void %0( ) nounwind
	ret i32 0
}
