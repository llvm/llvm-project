; All of these routines should be perform optimal load of constants.
; RUN: llvm-as < %s | llc -march=ppc32 | \
; RUN:   grep lis | count 5
; RUN: llvm-as < %s | llc -march=ppc32 | \
; RUN:   grep ori | count 3
; RUN: llvm-as < %s | llc -march=ppc32 | \
; RUN:   grep {li } | count 4

define i32 @f1() {
entry:
	ret i32 1
}

define i32 @f2() {
entry:
	ret i32 -1
}

define i32 @f3() {
entry:
	ret i32 0
}

define i32 @f4() {
entry:
	ret i32 32767
}

define i32 @f5() {
entry:
	ret i32 65535
}

define i32 @f6() {
entry:
	ret i32 65536
}

define i32 @f7() {
entry:
	ret i32 131071
}

define i32 @f8() {
entry:
	ret i32 2147483647
}

define i32 @f9() {
entry:
	ret i32 -2147483648
}
