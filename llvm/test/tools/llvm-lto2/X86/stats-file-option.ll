; REQUIRES: asserts

; RUN: llvm-as < %s > %t1.bc

; Try to save statistics to file.
; RUN: llvm-lto2 run %t1.bc -o %t.o -r %t1.bc,patatino,px -stats-file=%t2.stats
; RUN: FileCheck --input-file=%t2.stats %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @patatino() {
  fence seq_cst
  ret void
}

; CHECK: {
; CHECK: "asm-printer.EmittedInsts":
; CHECK: }


; Try to save statistics to an invalid file.
; RUN: not llvm-lto2 run %t1.bc -o %t.o -r %t1.bc,patatino,px \
; RUN:     -stats-file=%t2/foo.stats 2>&1 | FileCheck --check-prefix=ERROR %s
; ERROR: LTO::run failed: {{[Nn]}}o such file or directory
