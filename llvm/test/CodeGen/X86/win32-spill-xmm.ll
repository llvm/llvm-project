; RUN: llc -mcpu=generic -mtriple=i686-pc-windows-msvc -mattr=+sse < %s | FileCheck %s

; Check proper alignment of spilled vector

; CHECK-LABEL: spill_ok
; CHECK: subl    $32, %esp
; CHECK: movups  %xmm3, (%esp)
; CHECK: movl    $0, 16(%esp)
; CHECK: calll   _bar
define void @spill_ok(i32, ptr) {
entry:
  %2 = alloca i32, i32 %0
  %3 = load <16 x float>, ptr %1, align 64
  tail call void @bar(<16 x float> %3, i32 0) nounwind
  ret void
}

declare void @bar(<16 x float> %a, i32 %b)

; Check that proper alignment of spilled vector does not affect vargs

; CHECK-LABEL: vargs_not_affected
; CHECK: movl 28(%esp), %eax
define i32 @vargs_not_affected(<4 x float> %v, ptr %f, ...) {
entry:
  %ap = alloca ptr, align 4
  call void @llvm.va_start(ptr %ap)
  %argp.cur = load ptr, ptr %ap, align 4
  %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
  store ptr %argp.next, ptr %ap, align 4
  %0 = load i32, ptr %argp.cur, align 4
  call void @llvm.va_end(ptr %ap)
  ret i32 %0
}

declare void @llvm.va_start(ptr)

declare void @llvm.va_end(ptr)
