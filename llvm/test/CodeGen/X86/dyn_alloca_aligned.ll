; RUN: llc -mtriple=x86_64-linux < %s | FileCheck %s
define i32 @A(i32 %Size) {
; CHECK:  subq    %rcx, %rax
; CHECK:  andq    $-128, %rax
; CHECK:  movq    %rax, %rsp
  %A = alloca i8, i32 %Size, align 128
  %A_addr = ptrtoint ptr %A to i32
  ret i32 %A_addr
}
