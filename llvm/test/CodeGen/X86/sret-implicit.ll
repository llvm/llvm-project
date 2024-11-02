; RUN: llc -mtriple=x86_64-apple-darwin8 < %s | FileCheck %s --check-prefix=X64
; RUN: llc -mtriple=x86_64-pc-linux < %s | FileCheck %s --check-prefix=X64
; RUN: llc -mtriple=i686-pc-linux < %s | FileCheck %s --check-prefix=X86
; RUN: llc -mtriple=x86_64-apple-darwin8 -terminal-rule < %s | FileCheck %s --check-prefix=X64
; RUN: llc -mtriple=x86_64-pc-linux -terminal-rule < %s | FileCheck %s --check-prefix=X64

define void @sret_void(ptr sret(i32) %p) {
  store i32 0, ptr %p
  ret void
}

; X64-LABEL: sret_void
; X64-DAG: movq %rdi, %rax
; X64-DAG: movl $0, (%rdi)
; X64: retq

; X86-LABEL: sret_void
; X86: movl 4(%esp), %eax
; X86: movl $0, (%eax)
; X86: retl

define i256 @sret_demoted() {
  ret i256 0
}

; X64-LABEL: sret_demoted
; X64-DAG: movq %rdi, %rax
; X64-DAG: movq $0, (%rdi)
; X64: retq

; X86-LABEL: sret_demoted
; X86: movl 4(%esp), %eax
; X86: movl $0, (%eax)
; X86: retl
