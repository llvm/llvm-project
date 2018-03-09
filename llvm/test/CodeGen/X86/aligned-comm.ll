; RUN: llc < %s -mtriple=i686--
; RUN: llc < %s -mtriple=i386-apple-darwin10 | grep "array,16512,7"
; RUN: llc < %s -mtriple=i386-apple-darwin9 | grep "array,16512,7"

; Darwin 9+ should get alignment on common symbols.
@array = common global [4128 x i32] zeroinitializer, align 128
