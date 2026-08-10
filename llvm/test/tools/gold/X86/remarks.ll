; RUN: llvm-as %s -o %t.o

; RUN: not %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    -plugin-opt=-pass-remarks=inline %t.o -o %t2.o 2>&1 | FileCheck %s

; RUN: not %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:   %t.o -o %t2.o 2>&1 | FileCheck -allow-empty --check-prefix=NO-REMARK %s


; CHECK: 'f' inlined into '_start'
; NO-REMARK-NOT: inlined
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @bar()

define i32 @f() {
  %a = call i32 @bar()
  ret i32 %a
}

define i32 @_start() {
  %call = call i32 @f()
  ret i32 %call
}
