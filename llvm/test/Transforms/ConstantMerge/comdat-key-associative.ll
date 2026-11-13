; RUN: opt -passes=constmerge -S < %s | FileCheck %s

; @key is the key of a comdat group whose associative member @assoc (pinned by
; llvm.used) survives. Even though @key is locally unused, constmerge's
; dead-global cleanup must keep it: deleting the key would leave @assoc
; referencing a comdat whose key symbol no longer exists, which is illegal for
; COFF and aborts codegen in getComdatGVForCOFF.

target triple = "x86_64-pc-windows-msvc"

$key = comdat any
$initfn = comdat any

; CHECK: @key = internal thread_local global i32 0, comdat
@key = internal thread_local global i32 0, comdat, align 8
; CHECK: @assoc = internal constant ptr @initfn{{.*}}comdat($key)
@assoc = internal constant ptr @initfn, section ".CRT$XDU", comdat($key)
@llvm.used = appending global [1 x ptr] [ptr @assoc], section "llvm.metadata"

define internal void @initfn() comdat {
  ret void
}
