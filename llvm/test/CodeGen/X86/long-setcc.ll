; RUN: llc < %s -mtriple=i686-- | FileCheck %s

define i1 @t1(i64 %x) nounwind {
	%B = icmp slt i64 %x, 0
	ret i1 %B
}

; CHECK: t1
; CHECK: shrl
; CHECK-NOT: shrl
; CHECK: ret

define i1 @t2(i64 %x) nounwind {
	%tmp = icmp ult i64 %x, 4294967296
	ret i1 %tmp
}

; CHECK: t2
; CHECK: cmp
; CHECK-NOT: cmp
; CHECK: ret

define i1 @t3(i32 %x) nounwind {
	%tmp = icmp ugt i32 %x, -1
	ret i1 %tmp
}

; CHECK: t3
; CHECK: xor
; CHECK-NOT: xor
; CHECK: ret
