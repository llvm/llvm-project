; The COFF directives that name a symbol, such as .rva, are references to it.
; wine writes its RTTI tables with them.

; RUN: opt -o %t %s
; RUN: llvm-lto2 dump-symtab %t | FileCheck %s

; CHECK-DAG: D------X carrier
; CHECK-DAG: D------X table
; CHECK-DAG: DU-----X by_rva
; CHECK-DAG: DU-----X by_secrel
; CHECK-DAG: DU-----X by_symidx
; CHECK-DAG: DU-----X by_handler
; CHECK-DAG: D------X with_handler

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @carrier() {
  call void asm sideeffect ".globl table\0Atable:\0A.rva by_rva\0A.secrel32 by_secrel\0A.symidx by_symidx\0A.globl with_handler\0Awith_handler:\0A.seh_proc with_handler\0A.seh_handler by_handler, @except\0A.seh_endprologue\0Aret\0A.seh_endproc\0A", ""()
  ret void
}
