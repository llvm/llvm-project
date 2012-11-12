; RUN: llc < %s -mcpu=generic -mtriple=x86_64-pc-win32 | FileCheck %s

; Verify that the var arg parameters which are passed in registers are stored
; in home stack slots allocated by the caller and that AP is correctly
; calculated.
define void @average_va(i32 %count, ...) nounwind {
entry:
; CHECK: pushq
; CHECK: movq   %r9, 40(%rsp)
; CHECK: movq   %r8, 32(%rsp)
; CHECK: movq   %rdx, 24(%rsp)
; CHECK: leaq   24(%rsp), %rax

  %ap = alloca i8*, align 8                       ; <i8**> [#uses=1]
  %ap1 = bitcast i8** %ap to i8*                  ; <i8*> [#uses=1]
  call void @llvm.va_start(i8* %ap1)
  ret void
}

declare void @llvm.va_start(i8*) nounwind

; CHECK: f5:
; CHECK: pushq
; CHECK: leaq 56(%rsp),
define i8* @f5(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, ...) nounwind {
entry:
  %ap = alloca i8*, align 8
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  ret i8* %ap1
}

; CHECK: f4:
; CHECK: pushq
; CHECK: leaq 48(%rsp),
define i8* @f4(i64 %a0, i64 %a1, i64 %a2, i64 %a3, ...) nounwind {
entry:
  %ap = alloca i8*, align 8
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  ret i8* %ap1
}

; CHECK: f3:
; CHECK: pushq
; CHECK: leaq 40(%rsp),
define i8* @f3(i64 %a0, i64 %a1, i64 %a2, ...) nounwind {
entry:
  %ap = alloca i8*, align 8
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  ret i8* %ap1
}
