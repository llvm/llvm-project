; RUN: llvm-as %s -o %t.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=save-temps \
; RUN:    -shared %t.o -o %t2.o
; RUN: llvm-dis %t2.o.0.2.internalize.bc -o - | FileCheck %s

; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -m elf_x86_64 \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o -o %t2.o
; RUN: llvm-dis %t2.o -o - | FileCheck --check-prefix=NONAME %s

; CHECK: @GlobalValueName
; CHECK: @foo(i32 %in)
; CHECK: somelabel:
; CHECK:  %GV = load i32, ptr @GlobalValueName
; CHECK:  %add = add i32 %in, %GV
; CHECK:  ret i32 %add

; NONAME: @GlobalValueName
; NONAME: @foo(i32 %0)
; NONAME-NOT: somelabel:
; NONAME:  %2 = load i32, ptr @GlobalValueName
; NONAME:  %3 = add i32 %0, %2
; NONAME:  ret i32 %3

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@GlobalValueName = global i32 0

define i32 @foo(i32 %in) {
somelabel:
  %GV = load i32, ptr @GlobalValueName
  %add = add i32 %in, %GV
  ret i32 %add
}
