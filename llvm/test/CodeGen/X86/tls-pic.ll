; RUN: llc < %s -march=x86 -mtriple=i386-linux-gnu -relocation-model=pic | FileCheck -check-prefix=X32 %s
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-linux-gnu -relocation-model=pic | FileCheck -check-prefix=X64 %s

@i = thread_local global i32 15
@j = internal thread_local global i32 42
@k = internal thread_local global i32 42

define i32 @f1() {
entry:
	%tmp1 = load i32* @i
	ret i32 %tmp1
}

; X32: f1:
; X32:   leal i@TLSGD(,%ebx), %eax
; X32:   calll ___tls_get_addr@PLT

; X64: f1:
; X64:   leaq i@TLSGD(%rip), %rdi
; X64:   callq __tls_get_addr@PLT


@i2 = external thread_local global i32

define i32* @f2() {
entry:
	ret i32* @i
}

; X32: f2:
; X32:   leal i@TLSGD(,%ebx), %eax
; X32:   calll ___tls_get_addr@PLT

; X64: f2:
; X64:   leaq i@TLSGD(%rip), %rdi
; X64:   callq __tls_get_addr@PLT



define i32 @f3() {
entry:
	%tmp1 = load i32* @i		; <i32> [#uses=1]
	ret i32 %tmp1
}

; X32: f3:
; X32:   leal	i@TLSGD(,%ebx), %eax
; X32:   calll ___tls_get_addr@PLT

; X64: f3:
; X64:   leaq i@TLSGD(%rip), %rdi
; X64:   callq __tls_get_addr@PLT


define i32* @f4() nounwind {
entry:
	ret i32* @i
}

; X32: f4:
; X32:   leal	i@TLSGD(,%ebx), %eax
; X32:   calll ___tls_get_addr@PLT

; X64: f4:
; X64:   leaq i@TLSGD(%rip), %rdi
; X64:   callq __tls_get_addr@PLT


define i32 @f5() nounwind {
entry:
	%0 = load i32* @j, align 4
	%1 = load i32* @k, align 4
	%add = add nsw i32 %0, %1
	ret i32 %add
}

; X32:    f5:
; X32:      leal {{[jk]}}@TLSLDM(%ebx)
; X32: calll ___tls_get_addr@PLT
; X32: movl {{[jk]}}@DTPOFF(%e
; X32: addl {{[jk]}}@DTPOFF(%e

; X64:    f5:
; X64:      leaq {{[jk]}}@TLSLD(%rip), %rdi
; X64: callq	__tls_get_addr@PLT
; X64: movl {{[jk]}}@DTPOFF(%r
; X64: addl {{[jk]}}@DTPOFF(%r
