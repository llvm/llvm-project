; RUN: opt -passes=constmerge -S < %s | FileCheck %s

; @key is the key of a comdat group whose associative member @assoc (pinned by
; llvm.used) survives. @key is locally unused, so constmerge's dead-global
; cleanup deletes it -- leaving @assoc referencing a comdat whose key symbol no
; longer exists. That is illegal for COFF and aborts codegen in
; getComdatGVForCOFF.
;
; FIXME: @key must not be deleted while @assoc keeps the group alive.
; https://github.com/llvm/llvm-project/issues/199462

target triple = "x86_64-pc-windows-msvc"

$key = comdat any
$initfn = comdat any

; @key is wrongly removed, leaving the comdat($key) reference below dangling:
; CHECK-NOT: @key =
@key = internal thread_local global i32 0, comdat, align 8
; CHECK: @assoc = internal constant ptr @initfn{{.*}}comdat($key)
@assoc = internal constant ptr @initfn, section ".CRT$XDU", comdat($key)
@llvm.used = appending global [1 x ptr] [ptr @assoc], section "llvm.metadata"

define internal void @initfn() comdat {
  ret void
}
