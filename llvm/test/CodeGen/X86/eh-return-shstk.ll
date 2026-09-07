; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s --check-prefix=RET
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mattr=+shstk -verify-machineinstrs < %s | FileCheck %s --check-prefix=SHSTK
; RUN: llc -mtriple=x86_64-pc-windows-msvc -mattr=+shstk -verify-machineinstrs < %s | FileCheck %s --check-prefix=WIN64
; RUN: llc -mtriple=x86_64-uefi -mattr=+shstk -verify-machineinstrs < %s | FileCheck %s --check-prefix=UEFI64

define void @test(i64 %offset, ptr %handler) {
entry:
  call void @llvm.eh.return.i64(i64 %offset, ptr %handler)
  unreachable
}

; RET-LABEL: test:
; RET:       movq %rcx, %rsp
; RET-NEXT:  retq

; SHSTK-LABEL: test:
; SHSTK:      movq %rcx, %rsp
; SHSTK-NEXT: popq %rcx
; SHSTK-NEXT: jmpq *%rcx

; WIN64-LABEL: test:
; WIN64:       movq %rcx, %rsp
; WIN64-NEXT:  popq %rcx
; WIN64-NEXT:  rex64 jmpq *%rcx

; UEFI64-LABEL: test:
; UEFI64:       movq %rcx, %rsp
; UEFI64-NEXT:  popq %rcx
; UEFI64-NEXT:  rex64 jmpq *%rcx

declare void @llvm.eh.return.i64(i64, ptr)
