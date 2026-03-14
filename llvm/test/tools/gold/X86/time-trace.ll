; RUN: llvm-as %s -o %t.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 --plugin-opt=time-trace=%t2.json \
; RUN:    -shared %t.o -o /dev/null
; RUN: FileCheck --input-file %t2.json %s

; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 --plugin-opt=time-trace=%t2.json \
; RUN:    --plugin-opt=time-trace-granularity=250  \
; RUN:    -shared %t.o -o /dev/null
; RUN: FileCheck --input-file %t2.json %s

; RUN: not %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 --plugin-opt=time-trace=%t2.json \
; RUN:    --plugin-opt=time-trace-granularity=hello  \
; RUN:    -shared %t.o -o /dev/null 2> %t4.txt
; RUN: FileCheck --input-file %t4.txt %s --check-prefix=ERR

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f1() {
  ret void
}

define void @f2() {
  ret void
}

; CHECK: "traceEvents":
; ERR: Invalid time trace granularity: hello
