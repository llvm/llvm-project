; RUN: llc < %s -march=x86 -mattr=+sse | FileCheck %s
; RUN: llc < %s -march=x86 -mattr=+avx | FileCheck %s
; RUN: llc < %s -march=x86 -mattr=+sse -mattr=+prfchw | FileCheck %s -check-prefix=PRFCHW
; RUN: llc < %s -march=x86 -mcpu=slm | FileCheck %s -check-prefix=SLM
; RUN: llc < %s -march=x86 -mcpu=btver2 | FileCheck %s -check-prefix=PRFCHW
; RUN: llc < %s -march=x86 -mcpu=btver2 -mattr=-prfchw | FileCheck %s -check-prefix=NOPRFCHW

; rdar://10538297

define void @t(i8* %ptr) nounwind  {
entry:
; CHECK: prefetcht2
; CHECK: prefetcht1
; CHECK: prefetcht0
; CHECK: prefetchnta
; PRFCHW: prefetchw
; NOPRFCHW-NOT: prefetchw
; SLM: prefetchw
	tail call void @llvm.prefetch( i8* %ptr, i32 0, i32 1, i32 1 )
	tail call void @llvm.prefetch( i8* %ptr, i32 0, i32 2, i32 1 )
	tail call void @llvm.prefetch( i8* %ptr, i32 0, i32 3, i32 1 )
	tail call void @llvm.prefetch( i8* %ptr, i32 0, i32 0, i32 1 )
	tail call void @llvm.prefetch( i8* %ptr, i32 1, i32 3, i32 1 )
	ret void
}

declare void @llvm.prefetch(i8*, i32, i32, i32) nounwind 
