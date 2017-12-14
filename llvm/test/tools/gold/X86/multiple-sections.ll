; RUN: echo ".text.tin" > %t_order_lto.txt
; RUN: echo ".text._start" >> %t_order_lto.txt
; RUN: echo ".text.pat" >> %t_order_lto.txt
; RUN: llvm-as %s -o %t.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:     -m elf_x86_64 -o %t.exe %t.o \
; RUN:     --section-ordering-file=%t_order_lto.txt
; RUN: llvm-readobj -elf-output-style=GNU -t %t.exe | FileCheck %s

; Check that the order of the sections is tin -> _start -> pat.

; CHECK:      00000000004000d0     1 FUNC    LOCAL  DEFAULT    1 pat
; CHECK:      00000000004000b0     1 FUNC    LOCAL  DEFAULT    1 tin
; CHECK:      00000000004000c0    15 FUNC    GLOBAL DEFAULT    1 _start

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @pat() #0 {
  ret void
}

define void @tin() #0 {
  ret void
}

define i32 @_start() {
  call void @pat()
  call void @tin()
  ret i32 0
}

attributes #0 = { noinline optnone }
