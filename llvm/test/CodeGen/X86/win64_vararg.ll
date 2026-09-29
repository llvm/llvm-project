; RUN: llc < %s -mcpu=generic -mtriple=x86_64-pc-win32 | FileCheck %s

; Verify that the var arg parameters which are passed in registers are stored
; in home stack slots allocated by the caller and that AP is correctly
; calculated.
define void @average_va(i32 %count, ...) nounwind {
entry:
; CHECK: pushq
; CHECK-DAG: movq   %r9, 40(%rsp)
; CHECK-DAG: movq   %r8, 32(%rsp)
; CHECK-DAG: movq   %rdx, 24(%rsp)
; CHECK: leaq   24(%rsp), %rax

  %ap = alloca ptr, align 8                       ; <ptr> [#uses=1]
  call void @llvm.va_start(ptr %ap)
  ret void
}

declare void @llvm.va_start(ptr) nounwind
declare void @llvm.va_copy(ptr, ptr) nounwind

; CHECK-LABEL: f5:
; CHECK: pushq
; CHECK: leaq 56(%rsp),
define ptr @f5(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, ...) nounwind {
entry:
  %ap = alloca ptr, align 8
  call void @llvm.va_start(ptr %ap)
  ret ptr %ap
}

; CHECK-LABEL: f4:
; CHECK: pushq
; CHECK: leaq 48(%rsp),
define ptr @f4(i64 %a0, i64 %a1, i64 %a2, i64 %a3, ...) nounwind {
entry:
  %ap = alloca ptr, align 8
  call void @llvm.va_start(ptr %ap)
  ret ptr %ap
}

; CHECK-LABEL: f3:
; CHECK: pushq
; CHECK: leaq 40(%rsp),
define ptr @f3(i64 %a0, i64 %a1, i64 %a2, ...) nounwind {
entry:
  %ap = alloca ptr, align 8
  call void @llvm.va_start(ptr %ap)
  ret ptr %ap
}

; WinX86_64 uses char* for va_list. Verify that the correct amount of bytes
; are copied using va_copy.

; CHECK-LABEL: copy1:
; CHECK: subq $16
; CHECK: leaq 32(%rsp), [[REG_copy1:%[a-z]+]]
; CHECK: movq [[REG_copy1]], 8(%rsp)
; CHECK: movq [[REG_copy1]], (%rsp)
; CHECK: addq $16
; CHECK: ret
define void @copy1(i64 %a0, ...) nounwind {
entry:
  %ap = alloca ptr, align 8
  %cp = alloca ptr, align 8
  call void @llvm.va_start(ptr %ap)
  call void @llvm.va_copy(ptr %cp, ptr %ap)
  ret void
}

; CHECK-LABEL: copy4:
; CHECK: subq $16
; CHECK: leaq 56(%rsp), [[REG_copy4:%[a-z]+]]
; CHECK: movq [[REG_copy4]], 8(%rsp)
; CHECK: movq [[REG_copy4]], (%rsp)
; CHECK: addq $16
; CHECK: ret
define void @copy4(i64 %a0, i64 %a1, i64 %a2, i64 %a3, ...) nounwind {
entry:
  %ap = alloca ptr, align 8
  %cp = alloca ptr, align 8
  call void @llvm.va_start(ptr %ap)
  call void @llvm.va_copy(ptr %cp, ptr %ap)
  ret void
}

; CHECK-LABEL: arg4:
; CHECK: pushq
; va_start (optimized away as overwritten by va_arg)
; va_arg:
; CHECK: leaq 52(%rsp), [[REG_arg4_2:%[a-z]+]]
; CHECK: movq [[REG_arg4_2]], (%rsp)
; CHECK: movl 48(%rsp), %eax
; CHECK: popq
; CHECK: ret
define i32 @arg4(i64 %a0, i64 %a1, i64 %a2, i64 %a3, ...) nounwind {
entry:
  %ap = alloca ptr, align 8
  call void @llvm.va_start(ptr %ap)
  %tmp = va_arg ptr %ap, i32
  ret i32 %tmp
}

define void @sret_arg(ptr sret(i32) %agg.result, ptr nocapture readnone %format, ...) {
entry:
  %ap = alloca ptr
  call void @llvm.va_start(ptr %ap)
  %tmp = va_arg ptr %ap, i32
  store i32 %tmp, ptr %agg.result
  ret void
}
; CHECK-LABEL: sret_arg:
; CHECK: pushq
; CHECK: movq %rcx, %rax
; CHECK-DAG: movq %r9, 40(%rsp)
; CHECK-DAG: movq %r8, 32(%rsp)
; CHECK-DAG: leaq 36(%rsp), %[[sret:[^ ]*]]
; CHECK-DAG: movl %r8d, (%rax)
; CHECK-DAG: movq %[[sret]], (%rsp)
; CHECK: popq
; CHECK: retq
