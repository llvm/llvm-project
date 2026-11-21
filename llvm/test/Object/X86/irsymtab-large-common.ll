; The symtab must record the full 64-bit size; a 32-bit field would store 1.

; RUN: env LLVM_OVERRIDE_PRODUCER=producer opt -o %t %s
; RUN: env LLVM_OVERRIDE_PRODUCER=producer llvm-lto2 dump-symtab %t | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
source_filename = "irsymtab-large-common.ll"

; 4294967297 bytes = 4 GiB + 1 (low 32 bits are 1 if truncated)
@big = common global [4294967297 x i8] zeroinitializer, align 1

define void @use() {
  %p = getelementptr i8, ptr @big, i64 4294967296
  store i8 1, ptr %p, align 1
  ret void
}

; CHECK:      version: 4
; CHECK-NEXT: producer: producer
; CHECK-NEXT: target triple: x86_64-unknown-linux-gnu
; CHECK-NEXT: source filename: irsymtab-large-common.ll
; CHECK-NEXT: dependent libraries:
; CHECK-DAG:  D-C----- big
; CHECK-DAG:          size 4294967297 align 1
; CHECK-DAG:  D------X use
