; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s --check-prefix=RET
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mattr=+shstk -verify-machineinstrs < %s | FileCheck %s --check-prefix=SHSTK

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

declare void @llvm.eh.return.i64(i64, ptr)
