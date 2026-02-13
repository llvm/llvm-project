;; Test that we don't pad the x86-64 General Dynamic/Local Dynamic TLS code
;; sequence. It uses prefixes to allow linker relaxation. We need to disable
;; prefix or nop padding for it. For simplicity and consistency, disable for
;; Local Dynamic and 32-bit as well.
; RUN: llc -mtriple=i386 -relocation-model=pic -x86-branches-within-32B-boundaries < %s | FileCheck --check-prefixes=CHECK,X86 %s
; RUN: llc -mtriple=x86_64 -relocation-model=pic -x86-branches-within-32B-boundaries < %s | FileCheck --check-prefixes=CHECK,X64 %s

@gd = external thread_local global i32
@ld = internal thread_local global i32 0

define i32 @tls_get_addr() {
; CHECK-LABEL: tls_get_addr:
; CHECK: #noautopadding
; X86: leal gd@TLSGD(,%ebx), %eax
; X86: calll ___tls_get_addr@PLT
; X64: data16
; X64: leaq gd@TLSGD(%rip), %rdi
; X64: callq __tls_get_addr@PLT
; CHECK: #autopadding
; CHECK: #noautopadding
; X86: leal ld@TLSLDM(%ebx), %eax
; X86: calll ___tls_get_addr@PLT
; X64: leaq ld@TLSLD(%rip), %rdi
; X64: callq __tls_get_addr@PLT
; CHECK: #autopadding
  %1 = load i32, ptr @gd
  %2 = load i32, ptr @ld
  %3 = add i32 %1, %2
  ret i32 %3
}
