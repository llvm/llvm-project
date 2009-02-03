; RUN: llvm-as < %s | llc -march=x86 > %t
; RUN: grep btl %t | count 2
; RUN: grep cmov %t | count 2
; RUN: not grep test %t
; RUN: not grep set %t
; RUN: not grep j %t
; RUN: not grep cmovne %t
; RUN: not grep cmove %t

define i32 @foo(i32 %x, i32 %n, i32 %w, i32 %v) nounwind readnone {
entry:
	%0 = lshr i32 %x, %n		; <i32> [#uses=1]
	%1 = and i32 %0, 1		; <i32> [#uses=1]
	%toBool = icmp eq i32 %1, 0		; <i1> [#uses=1]
	%.0 = select i1 %toBool, i32 %v, i32 12		; <i32> [#uses=1]
	ret i32 %.0
}
define i32 @bar(i32 %x, i32 %n, i32 %w, i32 %v) nounwind readnone {
entry:
	%0 = lshr i32 %x, %n		; <i32> [#uses=1]
	%1 = and i32 %0, 1		; <i32> [#uses=1]
	%toBool = icmp eq i32 %1, 0		; <i1> [#uses=1]
	%.0 = select i1 %toBool, i32 12, i32 %v		; <i32> [#uses=1]
	ret i32 %.0
}
