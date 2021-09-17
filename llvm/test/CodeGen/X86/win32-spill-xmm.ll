; RUN: llc -mcpu=generic -mtriple=i686-pc-windows-msvc -mattr=+sse < %s | FileCheck %s

; Check proper alignment of spilled vector

; CHECK-LABEL: spill_ok
; CHECK: subl    $32, %esp
; CHECK: movaps  %xmm3, (%esp)
; CHECK: movl    $0, 16(%esp)
; CHECK: calll   _bar
define void @spill_ok(i32, <16 x float> *) {
entry:
  %2 = alloca i32, i32 %0
  %3 = load <16 x float>, <16 x float> * %1, align 64
  tail call void @bar(<16 x float> %3, i32 0) nounwind
  ret void
}

declare void @bar(<16 x float> %a, i32 %b)

; Check that proper alignment of spilled vector does not affect vargs

; CHECK-LABEL: vargs_not_affected
; CHECK: movl 28(%esp), %eax
define i32 @vargs_not_affected(<4 x float> %v, i8* %f, ...) {
entry:
  %ap = alloca i8*, align 4
  %0 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %0)
  %argp.cur = load i8*, i8** %ap, align 4
  %argp.next = getelementptr inbounds i8, i8* %argp.cur, i32 4
  store i8* %argp.next, i8** %ap, align 4
  %1 = bitcast i8* %argp.cur to i32*
  %2 = load i32, i32* %1, align 4
  call void @llvm.va_end(i8* %0)
  ret i32 %2
}

declare void @llvm.va_start(i8*)

declare void @llvm.va_end(i8*)
