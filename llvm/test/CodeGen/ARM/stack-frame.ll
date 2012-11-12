; RUN: llc < %s -march=arm
; RUN: llc < %s -march=arm | grep add | count 1

define void @f1() {
	%c = alloca i8, align 1
	ret void
}

define i32 @f2() {
	ret i32 1
}


