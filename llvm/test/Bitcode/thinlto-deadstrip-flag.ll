; REQUIRES: x86-registered-target
; RUN: opt -module-summary %s -o %t.o

; Ensure dead stripping performed flag is set on distributed index
; RUN: llvm-lto2 run %t.o -o %t.out -thinlto-distributed-indexes \
; RUN:		-r %t.o,glob,plx
; RUN: llvm-bcanalyzer -dump %t.o.thinlto.bc | FileCheck %s --check-prefix=WITHDEAD
; WITHDEAD: <FLAGS op0=1/>

; Ensure dead stripping performed flag is not set on distributed index
; when option used to disable dead stripping computation.
; RUN: llvm-lto2 run %t.o -o %t.out -thinlto-distributed-indexes \
; RUN:		-r %t.o,glob,plx -compute-dead=false
; RUN: llvm-bcanalyzer -dump %t.o.thinlto.bc | FileCheck %s --check-prefix=NODEAD
; NODEAD: <FLAGS op0=0/>

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@glob = global i32 0
