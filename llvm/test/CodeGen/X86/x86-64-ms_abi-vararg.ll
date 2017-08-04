; RUN: llc < %s -mcpu=generic -mtriple=x86_64-pc-linux-gnu | FileCheck %s

; Verify that the var arg parameters which are passed in registers are stored
; in home stack slots allocated by the caller and that AP is correctly
; calculated.
define win64cc void @average_va(i32 %count, ...) nounwind {
entry:
; CHECK: pushq
; CHECK: movq   %r9, 40(%rsp)
; CHECK: movq   %r8, 32(%rsp)
; CHECK: movq   %rdx, 24(%rsp)
; CHECK: leaq   24(%rsp), %rax

  %ap = alloca i8*, align 8                       ; <i8**> [#uses=1]
  %ap.0 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap.0)
  ret void
}

declare void @llvm.va_start(i8*) nounwind
declare void @llvm.va_copy(i8*, i8*) nounwind
declare void @llvm.va_end(i8*) nounwind

; CHECK-LABEL: f5:
; CHECK: pushq
; CHECK: leaq 56(%rsp),
define win64cc i8** @f5(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, ...) nounwind {
entry:
  %ap = alloca i8*, align 8
  %ap.0 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap.0)
  ret i8** %ap
}

; CHECK-LABEL: f4:
; CHECK: pushq
; CHECK: leaq 48(%rsp),
define win64cc i8** @f4(i64 %a0, i64 %a1, i64 %a2, i64 %a3, ...) nounwind {
entry:
  %ap = alloca i8*, align 8
  %ap.0 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap.0)
  ret i8** %ap
}

; CHECK-LABEL: f3:
; CHECK: pushq
; CHECK: leaq 40(%rsp),
define win64cc i8** @f3(i64 %a0, i64 %a1, i64 %a2, ...) nounwind {
entry:
  %ap = alloca i8*, align 8
  %ap.0 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap.0)
  ret i8** %ap
}

; WinX86_64 uses char* for va_list. Verify that the correct amount of bytes
; are copied using va_copy.

; CHECK-LABEL: copy1:
; CHECK: leaq 32(%rsp), [[REG_copy1:%[a-z]+]]
; CHECK: movq [[REG_copy1]], 8(%rsp)
; CHECK: movq [[REG_copy1]], (%rsp)
; CHECK: ret
define win64cc void @copy1(i64 %a0, ...) nounwind {
entry:
  %ap = alloca i8*, align 8
  %cp = alloca i8*, align 8
  %ap.0 = bitcast i8** %ap to i8*
  %cp.0 = bitcast i8** %cp to i8*
  call void @llvm.va_start(i8* %ap.0)
  call void @llvm.va_copy(i8* %cp.0, i8* %ap.0)
  ret void
}

; CHECK-LABEL: copy4:
; CHECK: leaq 56(%rsp), [[REG_copy4:%[a-z]+]]
; CHECK: movq [[REG_copy4]], 8(%rsp)
; CHECK: movq [[REG_copy4]], (%rsp)
; CHECK: ret
define win64cc void @copy4(i64 %a0, i64 %a1, i64 %a2, i64 %a3, ...) nounwind {
entry:
  %ap = alloca i8*, align 8
  %cp = alloca i8*, align 8
  %ap.0 = bitcast i8** %ap to i8*
  %cp.0 = bitcast i8** %cp to i8*
  call void @llvm.va_start(i8* %ap.0)
  call void @llvm.va_copy(i8* %cp.0, i8* %ap.0)
  ret void
}

; CHECK-LABEL: arg4:
; va_start (optimized away as overwritten by va_arg)
; va_arg:
; CHECK: leaq 52(%rsp), [[REG_arg4_2:%[a-z]+]]
; CHECK: movq [[REG_arg4_2]], (%rsp)
; CHECK: movl 48(%rsp), %eax
; CHECK: ret
define win64cc i32 @arg4(i64 %a0, i64 %a1, i64 %a2, i64 %a3, ...) nounwind {
entry:
  %ap = alloca i8*, align 8
  %ap.0 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap.0)
  %tmp = va_arg i8** %ap, i32
  ret i32 %tmp
}
