; RUN: llc < %s -mtriple=i386-linux-gnu -relocation-model=pic | FileCheck %s --check-prefix=X86
; RUN: llc < %s -mtriple=x86_64-linux-gnu -relocation-model=pic | FileCheck %s --check-prefix=X64

@i = thread_local global i32 15
@j = internal thread_local global i32 42
@k = internal thread_local global i32 42

define i32 @f1() {
entry:
	%tmp1 = load i32, ptr @i
	ret i32 %tmp1
}

; X86-LABEL: f1:
; X86:   leal i@TLSGD(,%ebx), %eax
; X86:   calll ___tls_get_addr@PLT

; X64-LABEL: f1:
; X64:   leaq i@TLSGD(%rip), %rdi
; X64:   callq __tls_get_addr@PLT


@i2 = external thread_local global i32

define ptr @f2() {
entry:
	ret ptr @i
}

; X86-LABEL: f2:
; X86:   leal i@TLSGD(,%ebx), %eax
; X86:   calll ___tls_get_addr@PLT

; X64-LABEL: f2:
; X64:   leaq i@TLSGD(%rip), %rdi
; X64:   callq __tls_get_addr@PLT



define i32 @f3() {
entry:
	%tmp1 = load i32, ptr @i		; <i32> [#uses=1]
	ret i32 %tmp1
}

; X86-LABEL: f3:
; X86:   leal	i@TLSGD(,%ebx), %eax
; X86:   calll ___tls_get_addr@PLT

; X64-LABEL: f3:
; X64:   leaq i@TLSGD(%rip), %rdi
; X64:   callq __tls_get_addr@PLT


define ptr @f4() nounwind {
entry:
	ret ptr @i
}

; X86-LABEL: f4:
; X86:   leal	i@TLSGD(,%ebx), %eax
; X86:   calll ___tls_get_addr@PLT

; X64-LABEL: f4:
; X64:   leaq i@TLSGD(%rip), %rdi
; X64:   callq __tls_get_addr@PLT


define i32 @f5() nounwind {
entry:
	%0 = load i32, ptr @j, align 4
	%1 = load i32, ptr @k, align 4
	%add = add nsw i32 %0, %1
	ret i32 %add
}

; X86-LABEL:    f5:
; X86:      leal {{[jk]}}@TLSLDM(%ebx)
; X86: calll ___tls_get_addr@PLT
; X86: movl {{[jk]}}@DTPOFF(%e
; X86: addl {{[jk]}}@DTPOFF(%e

; X64-LABEL:    f5:
; X64:      leaq {{[jk]}}@TLSLD(%rip), %rdi
; X64: callq	__tls_get_addr@PLT
; X64: movl {{[jk]}}@DTPOFF(%r
; X64: addl {{[jk]}}@DTPOFF(%r
