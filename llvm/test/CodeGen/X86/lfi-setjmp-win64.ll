; RUN: llc < %s -mtriple=x86_64_lfi | FileCheck %s --check-prefix=CHECK --implicit-check-not=%%r14 --implicit-check-not=%%r15
; RUN: llc < %s -mtriple=x86_64_lfi -filetype=obj -o /dev/null
; RUN: llc < %s -mtriple=x86_64_lfi -mattr=-sse | FileCheck %s --check-prefix=CHECK --implicit-check-not=%%r14 --implicit-check-not=%%r15
; RUN: llc < %s -mtriple=x86_64_lfi -mattr=-sse -filetype=obj -o /dev/null
; RUN: llc < %s -mtriple=x86_64_lfi -mattr=+egpr | FileCheck %s --check-prefixes=CHECK,APX --implicit-check-not=%%r14 --implicit-check-not=%%r15
; RUN: llc < %s -mtriple=x86_64_lfi -mattr=+egpr -filetype=obj -o /dev/null
; RUN: llc < %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,NATIVE

; The explicit Win64 calling convention can be used on non-Windows targets.
; LFI must exclude R14/R15 from its save lists while still preserving the
; Win64-only callee-saved registers, including R30/R31 with APX enabled.
; The native ABI must continue saving and restoring R14/R15.

@buf = internal global [5 x ptr] zeroinitializer, align 16

declare i32 @llvm.eh.sjlj.setjmp(ptr)

define win64cc i32 @test_setjmp_win64() {
; CHECK-LABEL: test_setjmp_win64:
; NATIVE:      pushq %r15
; NATIVE:      pushq %r14
; APX:         pushq %r31
; APX:         pushq %r30
; CHECK:       pushq %rsi
; CHECK:       pushq %rdi
; CHECK:       popq %rdi
; CHECK:       popq %rsi
; NATIVE:      popq %r14
; NATIVE:      popq %r15
; APX:         popq %r30
; APX:         popq %r31
entry:
  %result = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %result
}
