; RUN: opt -module-summary %s -o %t1.o
; RUN: opt -module-summary %p/Inputs/common_thinlto.ll -o %t2.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=save-temps \
; RUN:    --plugin-opt=thinlto \
; RUN:    -m elf_x86_64 \
; RUN:    -shared %t1.o %t2.o -o %t3

; RUN: llvm-dis %t1.o.2.internalize.bc -o - | FileCheck %s --check-prefix=INTERNALIZE
; We should not have internalized P, and it should still be common.
; INTERNALIZE: @P = common global

; RUN: llvm-dis %t1.o.4.opt.bc -o - | FileCheck %s --check-prefix=OPT
; bar should still exist (if we had internalized P it would look dead).
; OPT: @bar

; RUN: llvm-nm %t3 | FileCheck %s --check-prefix=NM
; NM: bar

source_filename = "common1.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@P = common global i8* (...)* null, align 8

define i32 @main() {
entry:
  store i8* (...)* bitcast (i8* ()* @bar to i8* (...)*), i8* (...)** @P, align 8
  %call = call i32 (...) @foo()
  ret i32 0
}

declare i32 @foo(...)

define internal i8* @bar() {
entry:
  ret i8* null
}
