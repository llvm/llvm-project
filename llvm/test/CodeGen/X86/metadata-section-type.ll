; RUN: llc -mtriple=x86_64-unknown-linux %s -o - | FileCheck %s

; Check that the section '.debug_llvm_dyndbg' is created without flags.
; FIXME: A more flexible approach would be to add metadata (like !exclude) to
; signal this, rather than checking for a specific section name.

@llvm.embedded.object = private constant [1 x i8] c"\00", section ".debug_llvm_dyndbg", align 1
@llvm.compiler.used = appending global [1 x ptr] [ptr @llvm.embedded.object], section "llvm.metadata"

; CHECK: .section .debug_llvm_dyndbg,"",@progbits
; CHECK-NEXT: .Lllvm.embedded.object:
; CHECK-NEXT: .zero  1
; CHECK-NEXT: .size .Lllvm.embedded.object, 1
