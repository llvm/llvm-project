; REQUIRES: asserts

; RUN: llvm-as -o %t.bc %s

; Try to save statistics to file.
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext -plugin-opt=stats-file=%t2.stats \
; RUN:    -m elf_x86_64 -r -o %t.o %t.bc
; RUN: FileCheck --input-file=%t2.stats %s

; CHECK: {
; CHECK: "asm-printer.EmittedInsts":
; CHECK: }


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo() {
  ret i32 10
}

; Try to save statistics to an invalid file.
; RUN: not %gold -plugin %llvmshlibdir/LLVMgold%shlibext -plugin-opt=stats-file=%t2/foo.stats \
; RUN:    -m elf_x86_64 -r -o %t.o %t.bc 2>&1 | FileCheck --check-prefix=ERROR %s
; ERROR: LLVM gold plugin: No such file or directory
