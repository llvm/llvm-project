; RUN: llc < %s -mtriple=i386-linux-musl -relocation-model=pic | FileCheck --check-prefixes=CHECK,X86 %s
; RUN: llc < %s -mtriple=x86_64-linux-musl -relocation-model=pic | FileCheck --check-prefixes=CHECK,X64 %s

@gd = thread_local global i32 0
@ld = internal thread_local global i32 0

define ptr @get_gd() {
entry:
; CHECK-LABEL: get_gd:
; X86: leal gd@TLSGD(%ebx), %eax
; X86: calll *___tls_get_addr@GOT(%ebx)

; X64: leaq gd@TLSGD(%rip), %rdi
; X64: callq *__tls_get_addr@GOTPCREL(%rip)
  ret ptr @gd
}

define ptr @get_ld() {
; FIXME: This function uses a single thread-local variable, we might want to fall back to general-dynamic.
; CHECK-LABEL: get_ld:
; X86: leal ld@TLSLDM(%ebx), %eax
; X86: calll *___tls_get_addr@GOT(%ebx)

; X64: leaq ld@TLSLD(%rip), %rdi
; X64: callq *__tls_get_addr@GOTPCREL(%rip)
  ret ptr @ld
}

!llvm.module.flags = !{!1}
!1 = !{i32 7, !"RtLibUseGOT", i32 1}
