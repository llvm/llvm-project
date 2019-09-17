; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target triple = "x86_64-pc-win32"

declare i64 @llvm.x86.flags.read.u64()
declare void @llvm.x86.flags.write.u64(i64)

define i64 @read_flags() {
entry:
  %flags = call i64 @llvm.x86.flags.read.u64()
  ret i64 %flags
}

; CHECK-LABEL: read_flags:
; CHECK:      pushq   %rbp
; CHECK:      .seh_pushreg %rbp
; CHECK:      movq    %rsp, %rbp
; CHECK:      .seh_setframe %rbp, 0
; CHECK:      .seh_endprologue
; CHECK-NEXT: pushfq
; CHECK-NEXT: popq    %rax
; CHECK-NEXT: popq    %rbp

define void @write_flags(i64 %arg) {
entry:
  call void @llvm.x86.flags.write.u64(i64 %arg)
  ret void
}

; CHECK-LABEL: write_flags:
; CHECK:      pushq   %rbp
; CHECK:      .seh_pushreg %rbp
; CHECK:      movq    %rsp, %rbp
; CHECK:      .seh_setframe %rbp, 0
; CHECK:      .seh_endprologue
; CHECK-NEXT: pushq   %rcx
; CHECK-NEXT: popfq
; CHECK-NEXT: popq    %rbp
