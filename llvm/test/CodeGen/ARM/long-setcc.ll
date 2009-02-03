; RUN: llvm-as < %s | llc -march=arm | grep cmp | count 1
; RUN: llvm-as < %s | llc -march=thumb | grep cmp | count 1


define i1 @t1(i64 %x) {
	%B = icmp slt i64 %x, 0
	ret i1 %B
}

define i1 @t2(i64 %x) {
	%tmp = icmp ult i64 %x, 4294967296
	ret i1 %tmp
}

define i1 @t3(i32 %x) {
	%tmp = icmp ugt i32 %x, -1
	ret i1 %tmp
}
