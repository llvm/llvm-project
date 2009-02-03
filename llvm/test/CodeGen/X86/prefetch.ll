; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse > %t
; RUN: grep prefetchnta %t
; RUN: grep prefetcht0 %t
; RUN: grep prefetcht1 %t
; RUN: grep prefetcht2 %t

define void @t(i8* %ptr) nounwind  {
entry:
	tail call void @llvm.prefetch( i8* %ptr, i32 0, i32 1 )
	tail call void @llvm.prefetch( i8* %ptr, i32 0, i32 2 )
	tail call void @llvm.prefetch( i8* %ptr, i32 0, i32 3 )
	tail call void @llvm.prefetch( i8* %ptr, i32 0, i32 0 )
	ret void
}

declare void @llvm.prefetch(i8*, i32, i32) nounwind 
